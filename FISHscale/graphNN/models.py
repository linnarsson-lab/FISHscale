import torchmetrics
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import tqdm
from pytorch_lightning import LightningModule
from FISHscale.graphNN.submodules import Classifier, PairNorm, DiffGroupNorm
from pyro.distributions import GammaPoisson
from torch.distributions import Gamma,Normal, Multinomial, kl_divergence as kl
import pyro
from pyro import distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.distributions import constraints
from pyro import poutine

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_mul_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_mul_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']
        
        pos_loss, neg_loss=  -F.logsigmoid(pos_score.sum(-1)), - F.logsigmoid(-neg_score.sum(-1))#.mean()
        loss = pos_loss + neg_loss
        #score = th.cat([pos_score, neg_score])
        #label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        #loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss#, pos_loss, neg_loss

class SAGELightning(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation=F.relu,
                 dropout=0.2,
                 lr=0.001,
                 supervised=False,
                 reference=0,
                 smooth=False,
                 device='cpu',
                 aggregator='attentional',
                 celltype_distribution=None,
                 ncells = None,
                 n_obs = None,
                 l_loc = None,
                 l_scale = None,
                 scale_factor = 1,
                 ):
        super().__init__()

        self.module = SAGE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, supervised,aggregator)
        self.lr = lr
        self.supervised= supervised
        self.loss_fcn = CrossEntropyLoss()
        self.kappa = 0
        self.reference=th.tensor(reference,dtype=th.float32, device=device)
        self.smooth = smooth
        self.n_hidden = n_hidden
        self.device= device

        if self.supervised:
            automatic_optimization = False
            self.l_loc = l_loc
            self.l_scale = l_scale
            self.train_acc = torchmetrics.Accuracy()
            self.kl = th.nn.KLDivLoss(reduction='sum')
            self.dist = celltype_distribution
            self.ncells = ncells

            A_factors_per_location, B_groups_per_location = 7 , 7
            self.factors_per_groups = A_factors_per_location / B_groups_per_location
            self.n_groups = self.module.n_hidden
            self.n_factors = self.module.n_classes
            self.n_obs= n_obs
            self.scale_factor = scale_factor
            self.alpha = 1
            self.device=device

    def model(self, x):
        pyro.module("decoder", self.module.decoder)
        _, pos,neg, mfgs = x
        pos_ids = pos.edges()[0]
        x = mfgs[1].dstdata['ngh']
        x= x[pos_ids,:]
        mfgs = [mfg.int() for mfg in mfgs]
        #batch_inputs = mfgs[0].srcdata['gene']

        hyp_alpha = th.tensor(9.0)
        hyp_beta = th.tensor(3.0)
        alpha_g_phi_hyp = pyro.sample("alpha_g_phi_hyp",
                dist.Gamma(hyp_alpha, hyp_beta),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([1, x.shape[1]]).to_event(2),
        )  # (self.n_batch, self.n_vars)

        #theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(x.shape[1]),
        #            constraint=constraints.positive)

        with pyro.plate("obs_plate", x.shape[0]),  poutine.scale(scale=self.scale_factor):

            zn_loc = x.new_ones(th.Size((x.shape[0], self.n_hidden)))*0
            zn_scale = x.new_ones(th.Size((x.shape[0], self.n_hidden)))*1
            zn = pyro.sample("zn",
                    dist.Normal(zn_loc, zn_scale).to_event(1)     
                )

            zm_loc = x.new_ones(th.Size((x.shape[0], self.n_hidden)))*0
            zm_scale = x.new_ones(th.Size((x.shape[0], self.n_hidden)))*1
            zm = pyro.sample("zm",
                    dist.Normal(zm_loc, zm_scale).to_event(1)     
                )

            l_scale = self.l_scale * x.new_ones(1)
            zl = pyro.sample("zl",
                    dist.LogNormal(self.l_loc, l_scale).to_event(1)     
                )
            
            z = zn*zm
            mu, gate_logits = self.module.decoder(z)
            mu = mu @ self.reference.T

            #nb_logits = (zl * mu + 6e-3).log() - (theta + 6e-3).log()
            gp_logits = zl * mu 
            '''x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta,
                                                       logits=nb_logits)'''

            x_dist =  dist.GammaPoisson(concentration=1/alpha_g_inverse, rate=1/(gp_logits*alpha_g_inverse)).to_event(1)
            pyro.sample("obs", 
                x_dist.to_event(1)
                ,obs=x
            )

    
    def guide(self,x):
        pyro.module("graph", self.module.encoder)
        pyro.module("molecule", self.module.encoder_molecule)
        _, pos, neg, mfgs = x
        pos_ids = pos.edges()[0]
        x = mfgs[1].dstdata['ngh']
        x= x[pos_ids,:]
        mfgs = [mfg.int() for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['gene']
        # register PyTorch module `decoder` with Pyro
    
        hyp_alpha= pyro.param('hyp_alpha',th.tensor(9.0))
        hyp_beta = pyro.param('hyp_beta', th.tensor(3.0))
        
        alpha_g_phi_hyp = pyro.sample("alpha_g_phi_hyp",
                dist.Gamma(hyp_alpha, hyp_beta),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([1, x.shape[1]]).to_event(2),
        )  # (self.n_batch, self.n_vars)
        

        with pyro.plate("obs_plate", x.shape[0]), poutine.scale(scale=self.scale_factor):            
            zn_loc, zn_scale = self.module.encoder(batch_inputs,
                                                    mfgs
                                                    )
            graph_loss = self.loss_fcn(zn_loc, pos, neg)#.mean()
            zm_loc, zm_scale, zl_loc, zl_scale = self.module.encoder_molecule(x)
            
            zn_loc,zn_scale = zn_loc[pos_ids,:], zn_scale[pos_ids,:],
            zm_loc, zm_scale = zm_loc[pos_ids,:],zm_scale[pos_ids,:]
            zl_loc, zl_scale =  zl_loc[pos_ids,:], zl_scale[pos_ids,:]

            zn = pyro.sample("zn", dist.Normal(zn_loc, th.sqrt(zn_scale)).to_event(1))
            zm = pyro.sample("zm", dist.Normal(zm_loc, th.sqrt(zm_scale)).to_event(1))
            zl = pyro.sample("zl", dist.LogNormal(zl_loc, th.sqrt(zl_scale)).to_event(1))

            pyro.factor("graph_loss", -self.alpha * graph_loss, has_rsample=False,)


    def validation_step(self,batch):
        _, pos, neg, mfgs = batch
        pos_ids = pos.edges()[0]
        x = mfgs[1].dstdata['ngh']
        x= x[pos_ids,:]
        mfgs = [mfg.int() for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['gene']
        zn_loc, zn_scale = self.module.encoder(batch_inputs,
                                                    mfgs
                                                    )

        val_loss = self.loss_fcn(zn_loc, pos, neg).mean()

        return val_loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.lr)
        return optimizer

class SAGE(nn.Module):
    def __init__(self, 
                in_feats, 
                n_hidden, 
                n_classes, 
                n_layers, 
                activation, 
                dropout,
                supervised,
                aggregator):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.supervised = supervised
        self.aggregator = aggregator
        #if self.supervised:
        #    n_hidden =n_classes

        self.encoder = Encoder(in_feats,
                                n_hidden,
                                n_classes,
                                n_layers,
                                supervised,
                                aggregator)
        
        self.encoder_molecule = EncoderMolecule(in_feats,
                                                    n_hidden
                                            )

        self.decoder = Decoder(in_feats, 
                                n_hidden,
                                n_classes,
                            )

    def inference(self, g, x, ngh,device, batch_size, num_workers):
            """
            Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
            g : the entire graph.
            x : the input of entire node set.
            The inference code is written in a fashion that it could handle any number of nodes and
            layers.
            """
            # During inference with sampling, multi-layer blocks are very inefficient because
            # lots of computations in the first few layers are repeated.
            # Therefore, we compute the representation of all nodes layer by layer.  The nodes
            # on each layer are of course splitted in batches.
            # TODO: can we standardize this?
            self.eval()
            for l, layer in enumerate(self.encoder.encoder_dict['GS']):
                y = th.zeros(g.num_nodes(), self.n_hidden) #if not self.supervised else th.zeros(g.num_nodes(), self.n_classes)
                p_class = th.zeros(g.num_nodes(), self.n_classes)

                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    th.arange(g.num_nodes()),#.to(g.device),
                    sampler,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    device=device,
                    num_workers=num_workers)

                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    
                    block = blocks[0]#.srcdata['gene']
                    block = block.int()
                    if l == 0:
                        h = th.log(x[input_nodes]+1).to(device)
                        #print('h',h.device)
                        #print(block.device)
                    else:
                        h = x[input_nodes].to(device)

                    if self.aggregator != 'attentional':
                        h = layer(block, h,)
                    else:
                        h = layer(block, h,).mean(1)
                        #h = self.encoder.encoder_dict['FC'][l](h)

                    if l == self.n_layers -1:
                        n = blocks[-1].dstdata['ngh']
                        h = self.encoder.gs_mu(h)
                        hm,_,_,_ = self.encoder_molecule(n)
                        h = h*hm
                        #h = self.encoder.softplus(h)
                        # then return a mean vector and a (positive) square root covariance
                        # each of size batch_size x z_dim
                        rate,_ = self.decoder(h)
                        p_class[output_nodes] = rate.cpu().detach()

                    #    h = self.mean_encoder(h)#, th.exp(self.var_encoder(h))+1e-4 )
                    y[output_nodes] = h.cpu().detach()#.numpy()
                x = y
        
            return y, p_class

class Encoder(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        supervised,
        aggregator,
        ):
        super().__init__()
        self.aggregator = aggregator
        layers = nn.ModuleList()
        if supervised:
            self.norm = F.normalize#DiffGroupNorm(n_hidden,n_classes,None) 
            #n_hidden = n_classes
        else:
            self.norm = F.normalize#DiffGroupNorm(n_hidden,20)

        for i in range(0,n_layers-1):
            if i > 0:
                in_feats = n_hidden
                x = 0.2
            else:
                x = 0

            if aggregator == 'attentional':
                layers.append(dglnn.GATConv(in_feats, 
                                            n_hidden, 
                                            num_heads=4,
                                            feat_drop=x,
                                            activation=F.relu,
                                            norm=self.norm,
                                            #allow_zero_in_degree=False
                                            ))

            else:
                layers.append(dglnn.SAGEConv(in_feats, 
                                            n_hidden, 
                                            aggregator_type=aggregator,
                                            #feat_drop=0.2,
                                            activation=F.relu,
                                            norm=self.norm,
                                            ))

        if aggregator == 'attentional':
            layers.append(dglnn.GATConv(n_hidden, 
                                        n_hidden, 
                                        num_heads=4, 
                                        feat_drop=0.2,
                                        activation=F.relu,
                                        #allow_zero_in_degree=False
                                        ))

        else:
            layers.append(dglnn.SAGEConv(n_hidden, 
                                            n_hidden, 
                                            aggregator_type=aggregator,
                                            feat_drop=0.2,
                                            activation=F.relu,
                                            norm=F.normalize
                                            ))

        self.encoder_dict = nn.ModuleDict({'GS': layers})
        self.gs_mu = nn.Linear(n_hidden, n_hidden)
        self.gs_var = nn.Linear(n_hidden, n_hidden)
        self.softplus = nn.Softplus()
    
    def forward(self,x, blocks=None):
        h = th.log(x+1)   
        for l, (layer, block) in enumerate(zip(self.encoder_dict['GS'], blocks)):
            if self.aggregator != 'attentional':
                h = layer(block, h,)
            else:
                h = layer(block, h,).mean(1)

        z_loc = self.gs_mu(h)
        z_scale = th.exp(self.gs_var(h))
        return z_loc, z_scale

    
class EncoderMolecule(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,

        ):
        super().__init__()

        self.softplus = nn.Softplus()
        self.fc = nn.Sequential(
                nn.Linear(in_feats, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU()      
            )
        self.mu = nn.Linear(n_hidden, n_hidden)
        self.var = nn.Linear(n_hidden, n_hidden)

        self.fc_l = nn.Sequential(
                nn.Linear(in_feats, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU()      
            )
        self.mu_l = nn.Linear(n_hidden, 1)
        self.var_l = nn.Linear(n_hidden, 1)
    
    def forward(self,x):
        x = th.log(x+1)   
        h = self.fc(x)
        z_loc = self.mu(h)
        z_scale = th.exp(self.var(h))

        h = self.fc_l(x)
        l_loc = self.mu_l(h)
        l_scale = th.exp(self.var_l(h))
        return z_loc, z_scale, l_loc, l_scale

class Decoder(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        
        ):
        super().__init__()

        self.fc = nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU()   
                )

        self.rate = nn.Linear(n_hidden, n_classes)
        self.gate = nn.Linear(n_hidden, in_feats)

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units

        hidden = self.fc(z)
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        mu = self.rate(hidden).softmax(dim=-1)
        gate = th.exp(self.gate(hidden))
        return mu, gate