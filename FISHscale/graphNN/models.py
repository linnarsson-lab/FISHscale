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
from scvi.nn import FCLayers

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

class SAGELightning(LightningModule):
    def __init__(self,
                 in_feats,
                 n_latent,
                 n_classes,
                 n_layers,
                 n_hidden=48,
                 dropout=0.1,
                 lr=0.0001,
                 supervised=False,
                 reference=0,
                 smooth=False,
                 device='cpu',
                 aggregator='attentional',
                 celltype_distribution=None,
                 ncells = None,
                 inference_type='deterministic',
                 l_loc = None,
                 l_scale = None,
                 scale_factor = 1,
                 ):
        super().__init__()

        self.module = SAGE(in_feats=in_feats, 
                            n_hidden=n_hidden,
                            n_latent=n_latent, 
                            n_classes=n_classes, 
                            n_layers=n_layers,
                            dropout=dropout, 
                            supervised=supervised,
                            aggregator= aggregator)

        self.lr = lr
        self.supervised= supervised
        self.loss_fcn = CrossEntropyLoss()
        self.kappa = 0
        self.reference=th.tensor(reference,dtype=th.float32, device=device)
        self.smooth = smooth
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.inference_type = inference_type

        if self.inference_type == 'VI':
            self.automatic_optimization = False
            self.svi = PyroOptWrap(model=self.model,
                    guide=self.guide,
                    optim=pyro.optim.Adam({"lr": self.lr}),
                    loss=pyro.infer.Trace_ELBO())

        if self.supervised:
            #self.automatic_optimization = False
            self.l_loc = l_loc
            self.l_scale = l_scale
            self.train_acc = torchmetrics.Accuracy()
            self.kl = th.nn.KLDivLoss(reduction='sum')
            self.dist = celltype_distribution
            self.ncells = ncells
            self.scale_factor = scale_factor
            self.alpha = 1


    def model(self, x):
        pyro.module("decoder", self.module.decoder)
        _, pos,neg, mfgs = x
        pos_ids = pos.edges()[0]
        x = mfgs[1].dstdata['ngh']
        x= x[pos_ids,:]
        mfgs = [mfg.int() for mfg in mfgs]
        #batch_inputs = mfgs[0].srcdata['gene']

        with pyro.plate("obs_plate", x.shape[0]),  poutine.scale(scale=self.scale_factor):

            zn_loc = x.new_ones(th.Size((x.shape[0], self.n_latent)))*0
            zn_scale = x.new_ones(th.Size((x.shape[0], self.n_latent)))*1
            zn = pyro.sample("zn",
                    dist.Normal(zn_loc, zn_scale).to_event(1)     
                )

            zm_loc = x.new_ones(th.Size((x.shape[0], self.n_latent)))*0
            zm_scale = x.new_ones(th.Size((x.shape[0], self.n_latent)))*1
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
            mu = zl * mu 

            alpha = 1/(th.exp(gate_logits).pow(2)) + 1e-6
            rate = alpha/mu

            x_dist =  dist.GammaPoisson(concentration=alpha, rate=rate).to_event(1)
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
    
        with pyro.plate("obs_plate", x.shape[0]), poutine.scale(scale=self.scale_factor):            
            zn_loc, zn_scale = self.module.encoder(batch_inputs,
                                                    mfgs
                                                    )
            graph_loss = self.loss_fcn(zn_loc, pos, neg)#.mean()
            zm_loc, zm_scale, zl_loc, zl_scale = self.module.encoder_molecule(x)
            
            zn_loc,zn_scale = zn_loc[pos_ids,:], zn_scale[pos_ids,:],
            zm_loc, zm_scale = zm_loc[pos_ids,:],zm_scale[pos_ids,:]
            zl_loc, zl_scale =  zl_loc[pos_ids,:], zl_scale[pos_ids,:]
            #za_loc, za_scale =  za_loc[pos_ids,:], za_scale[pos_ids,:]

            zn = pyro.sample("zn", dist.Normal(zn_loc, th.sqrt(zn_scale)).to_event(1))
            zm = pyro.sample("zm", dist.Normal(zm_loc, th.sqrt(zm_scale)).to_event(1))
            zl = pyro.sample("zl", dist.LogNormal(zl_loc, th.sqrt(zl_scale)).to_event(1))
            #za = pyro.sample("za", dist.Normal(za_loc, th.sqrt(za_scale)).to_event(1))
            pyro.factor("graph_loss", self.alpha * graph_loss, has_rsample=False,)

    def training_step(self, batch, batch_idx):
        if self. inference_type == 'VI':
            loss = self.svi.step(batch)
            self.log('train_loss',loss, prog_bar=True)

        else:
            self.reference = self.reference.to(self.device)
            _, pos, neg, mfgs = batch
            pos_ids = pos.edges()[0]
            x = mfgs[1].dstdata['ngh']
            x= x[pos_ids,:]
            mfgs = [mfg.int() for mfg in mfgs]
            batch_inputs = mfgs[0].srcdata['gene']
            zn_loc, _ = self.module.encoder(batch_inputs,mfgs)
            graph_loss = self.loss_fcn(zn_loc, pos, neg).mean()

            if self.supervised:
                zm_loc, _, zl_loc, _ = self.module.encoder_molecule(x)

                zn_loc = zn_loc[pos_ids,:]
                zm_loc = zm_loc[pos_ids,:]
                zl_loc =  zl_loc[pos_ids,:]

                z = zn_loc*zm_loc
                px_scale,px_r, px_dropout = self.module.decoder(z)
                px_scale = px_scale @ self.reference.T
                px_rate = th.exp(zl_loc) * px_scale

                alpha = 1/(th.exp(px_r).pow(2)) + 1e-6
                rate = alpha/px_rate
                NB = GammaPoisson(concentration=alpha,rate=rate)#.log_prob(local_nghs).mean(axis=-1).mean()
                nb_loss = -NB.log_prob(x).mean(axis=-1).mean()
            
                # Regularize by local nodes
                # Add Predicted same class nodes together.
                if type(self.dist) != type(None):
                    #option 2
                    p = th.ones(px_scale.shape[0]) @ px_scale
                    p = th.log(p/p.sum())
                    loss_dist = self.kl(p,self.dist.to(self.device)).sum()
                else:
                    loss_dist = 0

                loss = graph_loss + nb_loss + loss_dist
                self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
                self.log('Loss Dist', loss_dist, prog_bar=True, on_step=True, on_epoch=True)
                self.log('nb_loss', nb_loss, prog_bar=True, on_step=True, on_epoch=True)
                self.log('Graph Loss', graph_loss, prog_bar=False, on_step=True, on_epoch=False)

            else:
                loss = graph_loss
                self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.lr)
        lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer,)
        scheduler = {
            'scheduler': lr_scheduler, 
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'train_loss'
        }
        return [optimizer],[scheduler]

    def validation_step(self,batch, batch_idx):
        if self. inference_type == 'VI':
                    loss = self.svi.step(batch)
                    self.log('train_loss',loss, prog_bar=True)

        else:
            self.reference = self.reference.to(self.device)
            _, pos, neg, mfgs = batch
            pos_ids = pos.edges()[0]
            x = mfgs[1].dstdata['ngh']
            x= x[pos_ids,:]
            mfgs = [mfg.int() for mfg in mfgs]
            batch_inputs = mfgs[0].srcdata['gene']
            zn_loc, _ = self.module.encoder(batch_inputs,mfgs)
            graph_loss = self.loss_fcn(zn_loc, pos, neg).mean()

            if self.supervised:
                zm_loc, _, zl_loc, _ = self.module.encoder_molecule(x)

                zn_loc = zn_loc[pos_ids,:]
                zm_loc = zm_loc[pos_ids,:]
                zl_loc =  zl_loc[pos_ids,:]

                z = zn_loc*zm_loc
                px_scale,px_r, px_dropout = self.module.decoder(z)
                px_scale = px_scale @ self.reference.T
                px_rate = th.exp(zl_loc) * px_scale

                alpha = 1/(th.exp(px_r).pow(2)) + 1e-6
                rate = alpha/px_rate
                NB = GammaPoisson(concentration=alpha,rate=rate)#.log_prob(local_nghs).mean(axis=-1).mean()
                nb_loss = -NB.log_prob(x).mean(axis=-1).mean()
            
                # Regularize by local nodes
                # Add Predicted same class nodes together.
                if type(self.dist) != type(None):
                    #option 2
                    p = th.ones(px_scale.shape[0]) @ px_scale
                    p = th.log(p/p.sum())
                    loss_dist = self.kl(p,self.dist.to(self.device)).sum()
                else:
                    loss_dist = 0

                loss = graph_loss + nb_loss + loss_dist

            else:
                loss = graph_loss
        return loss


class PyroOptWrap(pyro.infer.SVI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def state_dict(self,):
        return {}

class SAGE(nn.Module):
    def __init__(self, 
                in_feats, 
                n_hidden,
                n_latent,
                n_classes, 
                n_layers, 
                dropout,
                supervised,
                aggregator):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_classes = n_classes
        self.supervised = supervised
        self.aggregator = aggregator

        self.encoder = Encoder(in_feats=in_feats,
                                n_hidden=n_hidden,
                                n_latent=n_latent,
                                n_layers=n_layers,
                                supervised=supervised,
                                aggregator=aggregator,
                                dropout= dropout,
                                )
        
        self.encoder_molecule = EncoderMolecule(in_feats=in_feats,
                                                    n_hidden=n_hidden,
                                                    n_latent= n_latent,
                                            )

        self.decoder = Decoder(in_feats=in_feats,
                                n_hidden=n_hidden,
                                n_latent= n_latent,
                                n_classes= n_classes,
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
                if l == self.n_layers - 1:
                    y = th.zeros(g.num_nodes(), self.n_latent) #if not self.supervised else th.zeros(g.num_nodes(), self.n_classes)
                else:
                    y = th.zeros(g.num_nodes(), self.n_hidden)
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
                        px_scale, px_r, px_dropout = self.decoder(h)
                        p_class[output_nodes] = px_scale.cpu().detach()

                    #    h = self.mean_encoder(h)#, th.exp(self.var_encoder(h))+1e-4 )
                    y[output_nodes] = h.cpu().detach()#.numpy()
                x = y
        
            return y, p_class

class Encoder(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_latent,
        n_layers,
        supervised,
        aggregator,
        dropout,
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
                                        feat_drop=dropout,
                                        activation=F.relu,
                                        #allow_zero_in_degree=False
                                        ))

        else:
            layers.append(dglnn.SAGEConv(n_hidden, 
                                            n_hidden, 
                                            aggregator_type=aggregator,
                                            feat_drop=dropout,
                                            activation=F.relu,
                                            norm=F.normalize
                                            ))

        self.encoder_dict = nn.ModuleDict({'GS': layers})
        self.gs_mu = nn.Linear(n_hidden, n_latent)
        self.gs_var = nn.Linear(n_hidden, n_latent)
    
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
        n_hidden=64,
        n_latent=24,

        ):
        super().__init__()

        self.softplus = nn.Softplus()
        self.fc = FCLayers(in_feats, n_hidden)
        self.mu = nn.Linear(n_hidden, n_latent)
        self.var = nn.Linear(n_hidden, n_latent)

        self.fc_l =FCLayers(in_feats, n_hidden)   
        self.mu_l = nn.Linear(n_hidden, 1)
        self.var_l = nn.Linear(n_hidden, 1)
    
    def forward(self,x):
        x = th.log(x+1)   
        h = self.fc(x)
        z_loc = self.mu(h)
        z_scale = th.exp(self.var(h))

        hl = self.fc_l(x)
        l_loc = self.mu_l(hl)
        l_scale = th.exp(self.var_l(hl))
        return z_loc, z_scale, l_loc, l_scale

class Decoder(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_latent,
        n_classes,
        
        ):
        super().__init__()

        self.fc = FCLayers(n_latent,n_hidden)

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_classes),
            nn.Softmax(dim=-1)
            )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, in_feats)
        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, in_feats)

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        px = self.fc(z)
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        #   # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px)
        return px_scale, px_r, px_dropout