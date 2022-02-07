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
        
        pos_loss, neg_loss=  -F.logsigmoid(pos_score.sum(-1)).mean(), - F.logsigmoid(-neg_score.sum(-1)).mean()
        loss = pos_loss + neg_loss
        #score = th.cat([pos_score, neg_score])
        #label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        #loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss, pos_loss, neg_loss

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
                 ):
        super().__init__()

        self.module = SAGE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, supervised,aggregator)
        self.lr = lr
        self.supervised= supervised
        self.loss_fcn = CrossEntropyLoss()
        self.kappa = 0
        self.reference=th.tensor(reference,dtype=th.float32)
        self.smooth = smooth
        self.n_hidden = n_hidden

        if self.supervised:
            automatic_optimization = False
            self.train_acc = torchmetrics.Accuracy()
            self.kl = th.nn.KLDivLoss(reduction='sum')
            self.dist = celltype_distribution
            self.ncells = ncells

            A_factors_per_location, B_groups_per_location = 7 , 7
            self.factors_per_groups = A_factors_per_location / B_groups_per_location
            self.n_groups = self.module.n_hidden
            self.n_factors = self.module.n_classes
            self.n_obs= n_obs

    def model(self, x):
        pyro.module("decoder", self.module.decoder)
        _, nids, mfgs = x
        x = mfgs[1].dstdata['ngh']
        mfgs = [mfg.int() for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['gene']

        hyp_alpha = th.tensor(9.0)
        hyp_beta = th.tensor(3.0)
        alpha_g_phi_hyp = pyro.sample("alpha_g_phi_hyp",
                dist.Gamma(hyp_alpha, hyp_beta),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([1, x.shape[1]]).to_event(2),
        )  # (self.n_batch, self.n_vars)

        with pyro.plate("obs_plate", x.shape[0]):

            z_loc = x.new_zeros(th.Size((x.shape[0], self.n_hidden)))
            z_scale = x.new_ones(th.Size((x.shape[0], self.n_hidden)))
            z = pyro.sample("z",
                    dist.Normal(z_loc, z_scale).to_event(1)     
                )
            #z_tmp = dist.Normal(z_loc, z_scale)
            #print('z',z_tmp.shape, z_tmp.batch_shape, z_tmp.event_shape)
            rate, shape = self.module.decoder(z)
            mu = rate @ self.reference.T
            alpha = 1/alpha_g_inverse.pow(2)
            rate = alpha/mu

            pyro.sample("obs", 
                dist.GammaPoisson(concentration=alpha, rate=rate).to_event(1)
                ,obs=x
            )

            #p_tmp = dist.GammaPoisson(concentration=alpha, rate=rate)
            #print('p_tmp',p_tmp.shape, p_tmp.batch_shape, p_tmp.event_shape)

    
    def guide(self,x):
        pyro.module("graph_predict", self.module.encoder)
        _, nids, mfgs = x
        x = mfgs[1].dstdata['ngh']
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

        with pyro.plate("obs_plate", x.shape[0]):
            #z_loc, z_scale = self.module(mfgs, batch_inputs_u)
            z_loc, z_scale = self.module.encoder(batch_inputs,mfgs)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1)
            )

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int() for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['gene']
        batch_pred = self.module(mfgs, batch_inputs)
        return batch_pred

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

        self.decoder = Decoder(n_classes,n_hidden)

    def inference(self, g, x, device, batch_size, num_workers):
            """
            Inference with the GraphSAGE model on full neighbors (i.e. without 
            neighbor sampling).
            
            This is used to reduce computing of all neighbors for more than 1-hop 
            aggregation which generates a large computing graph in memory and 
            slows down generating the embedding.

            g : the entire graph.
            x : the input of entire node set.
            The inference code is written in a fashion that it could handle any number of nodes and
            layers.
            """
            self.eval()
            for l, layer in enumerate(self.encoder.graphsage):

                y = th.zeros(g.num_nodes(), self.n_hidden) #if not self.supervised else th.zeros(g.num_nodes(), self.n_classes)
                p = th.zeros(g.num_nodes(), self.n_classes)


                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    th.arange(g.num_nodes()),#.to(g.device),
                    sampler,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers)

                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    block = blocks[0]#.srcdata['gene']
                    block = block.int()
                    if l == 0:
                        h = th.log(x[input_nodes]+1)#.to(device)
                    else:
                        h = x[input_nodes]

                    if self.aggregator != 'attentional':
                        h = layer(block, h,)
                    else:
                        h = layer(block, h,).mean(1)
                        #h = self.encoder.encoder_dict['FC'][l](h)

                    if l == self.n_layers -1:
                        h = self.encoder.fc(h)
                        h = self.encoder.softplus(h)
                        h = self.encoder.fc21(h)
                        mu,_ = self.decoder(h)
                        p[output_nodes] = mu.cpu().detach()
                    
                    y[output_nodes] = h.cpu().detach()#.numpy()
                x = y
        
            return y,p

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
                                        #activation=F.relu,
                                        #allow_zero_in_degree=False
                                        ))

        else:
            layers.append(dglnn.SAGEConv(n_hidden, 
                                            n_hidden, 
                                            aggregator_type=aggregator,
                                            feat_drop=0.2,
                                            #activation=F.relu,
                                            #norm=F.normalize
                                            ))

        self.graphsage = layers
        self.fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU()      
            )

        self.fc21 = nn.Linear(n_hidden, n_hidden)
        self.fc22 = nn.Linear(n_hidden, n_hidden)
        self.softplus = nn.Softplus()
    
    def forward(self,x, blocks=None):
        h = th.log(x+1)   
        for l, (layer, block) in enumerate(zip(self.graphsage, blocks)):
            if self.aggregator != 'attentional':
                h = layer(block, h,)
            else:
                h = layer(block, h,).mean(1)
        h = self.fc(h)
        h = self.softplus(h)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(h)
        z_scale = th.exp(self.fc22(h))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        
        ):
        super().__init__()

        self.fc = nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU()      
                )
        self.softplus =  nn.Softplus()
    
        self.rate = nn.Linear(n_hidden, in_feats)
        self.shape = nn.Linear(n_hidden, in_feats)

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        rate = self.softplus(self.rate(hidden))
        shape = self.softplus(self.shape(hidden))
        return rate, shape