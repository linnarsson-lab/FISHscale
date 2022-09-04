from matplotlib.pyplot import get
import torchmetrics
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import tqdm
from pytorch_lightning import LightningModule
from FISHscale.graphNN.submodules import Classifier
from pyro.distributions import GammaPoisson, Poisson
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
        
        pos_loss, neg_loss=  -F.logsigmoid(pos_score.sum(-1)), - F.logsigmoid(-neg_score.sum(-1))
        neg_loss = neg_loss.reshape([pos_score.shape[0], int(neg_score.shape[0]/pos_score.shape[0])])#.mean(axis=1)
        loss = pos_loss + neg_loss.mean(axis=1)
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
                 n_hidden=64,
                 dropout=0.1,
                 lr=0.001,
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
                 warmup_factor = 1,
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

        self.automatic_optimization = False
        if self.supervised:
            self.l_loc = l_loc
            self.l_scale = l_scale
            self.train_acc = torchmetrics.Accuracy()
            self.kl = th.nn.KLDivLoss(reduction='sum')
            self.dist = celltype_distribution
            self.ncells = ncells
            self.scale_factor = scale_factor
            self.alpha = 1
            self.warmup_counter = 0
            self.warmup_factor=warmup_factor


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
            px_scale, px_r, px_dropout = self.module.decoder(z)
            px_scale = px_scale @ self.reference.T.to(self.device)
            px_rate = zl * px_scale +1e-6

            alpha = 1/(th.exp(px_r)) +1e-6
            rate = alpha/px_rate

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
            self.log('Graph Loss', self.alpha*graph_loss.mean(), prog_bar=False, on_step=True, on_epoch=False)

    

    def training_step(self, batch, batch_idx):
        if self. inference_type == 'VI':
            loss = self.svi.step(batch)

        else:
            self.reference = self.reference.to(self.device)
            _, pos, neg, mfgs = batch
            pos_ids = pos.edges()[0]
            mfgs = [mfg.int() for mfg in mfgs]
            batch_inputs = mfgs[0].srcdata['gene']
            zn_loc, _ = self.module.encoder(batch_inputs,mfgs)
            graph_loss = self.loss_fcn(zn_loc, pos, neg).mean()
            opt_g, opt_nb= self.optimizers()
            opt_g.zero_grad()
            self.manual_backward(graph_loss)
            opt_g.step()

            if self.supervised:
                x = mfgs[-1].dstdata['ngh']
                x= x[pos_ids,:]
                zm_loc, _, zl_loc, _ = self.module.encoder_molecule(x)
                zn_loc = zn_loc[pos_ids,:]
                zm_loc = zm_loc[pos_ids,:]
                zl_loc =  zl_loc[pos_ids,:]

                new_ref = self.reference.T
                #th.distributions.Multinomial(
                #total_count=int(x.sum(axis=1).mean()), 
                #probs=self.reference,
                #).sample().to(self.device)
                #new_ref = new_ref.T/new_ref.sum(axis=1)
                z = zn_loc.detach()*zm_loc
                #px_scale_c, px_r, px_l = self.module.decoder(z)
                px_scale = z @ self.module.encoder_molecule.module2celltype
                px_scale_c = px_scale.softmax(dim=-1)
                #px_r = self.module.encoder_molecule.dispersion

                #px_rate = th.exp(zl_loc) * (px_scale_c @ self.reference.T)
                px_scale = px_scale_c @ new_ref
                #px_rate = th.exp(zl_loc) * (px_scale) +1e-6

                #alpha = 1/(th.exp(px_r)) + 1e-6
                #rate = alpha/px_rate
                #NB = GammaPoisson(concentration=alpha,rate=rate)#.log_prob(local_nghs).mean(axis=-1).mean()
                #NB = Poisson(px_rate)#.log_prob(local_nghs).mean(axis=-1).mean()
                #nb_loss = -NB.log_prob(x).sum(axis=-1).mean()
                # Regularize by local nodes
                # Add Predicted same class nodes together.

                nb_loss = -F.cosine_similarity(px_scale, x,dim=1).mean()#/x.shape[0]
                nb_loss += -F.cosine_similarity(px_scale, x,dim=0).mean()#/x.shape[0]
                #entropy_regularizer = (th.log(px_scale_c) * px_scale_c).sum()
                #nb_loss += entropy_regularizer
                #nb_loss = -self.lambda_r * (torch.log(M_probs) * M_probs).sum()

                if type(self.dist) != type(None):
                    #option 2
                    p = th.ones(px_scale_c.shape[0],device=self.device) @ px_scale_c
                    p = th.log(p/p.sum())
                    loss_dist = self.kl(p,self.dist.to(self.device)).sum()
                else:
                    loss_dist = 0

                uns_warmup = min(1,self.warmup_counter/self.warmup_factor)
                self.warmup_counter += 1
                
                loss = nb_loss + loss_dist #+ graph_loss
                #self.losses.append(loss)
                opt_nb.zero_grad()
                self.manual_backward(loss)
                opt_nb.step()

                self.log('train_loss', 
                    loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=zn_loc.shape[0])
                self.log('Loss Dist',
                    loss_dist, prog_bar=True, on_step=True, on_epoch=True, batch_size=zn_loc.shape[0])
                self.log('nb_loss', 
                    nb_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=zn_loc.shape[0])
                self.log('Graph Loss', 
                    graph_loss, prog_bar=False, on_step=True, on_epoch=False,batch_size= zn_loc.shape[0])

            else:
                loss = graph_loss
                self.log('train_loss', 
                    loss, prog_bar=True, on_step=True, on_epoch=True,batch_size=zn_loc.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer_graph = th.optim.Adam(self.module.encoder.parameters(), lr=self.lr)
        optimizer_nb = th.optim.Adam(self.module.encoder_molecule.parameters(), lr=0.01)
        lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nb,)
        scheduler = {
            'scheduler': lr_scheduler, 
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'train_loss'
        }
        return [optimizer_graph, optimizer_nb],[scheduler]

    def validation_step(self,batch, batch_idx):
        pass

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
                                                    n_classes=n_classes,
                                                    n_hidden=n_hidden,
                                                    n_latent= n_latent,
                                            )

        self.decoder = Decoder(in_feats=in_feats,
                                n_hidden=n_hidden,
                                n_latent= n_latent,
                                n_classes= n_classes,
                        )

    def inference(self, g,device, batch_size, num_workers):
            """
            Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
            g : the entire graph.
            The inference code is written in a fashion that it could handle any number of nodes and
            layers.
            """
            self.eval()

            g.ndata['h'] = th.log(g.ndata['gene']+1)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
            dataloader = dgl.dataloading.NodeDataLoader(
                    g, th.arange(g.num_nodes()).to(g.device), sampler, device=device,
                    batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                    persistent_workers=(num_workers > 0))
            
            for l, layer in enumerate(self.encoder.encoder_dict['GS']):
                if l == self.n_layers - 1:
                    y = th.zeros(g.num_nodes(), self.n_latent) #if not self.supervised else th.zeros(g.num_nodes(), self.n_classes)
                else:
                    if self.aggregator == 'attentional':
                        y = th.zeros(g.num_nodes(), self.n_hidden*4)
                    else:
                        y = th.zeros(g.num_nodes(), self.n_hidden)
                
                if self.supervised:
                    p_class = th.zeros(g.num_nodes(), self.n_classes)
                else:
                    p_class = None

                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    x = blocks[0].srcdata['h']
                    if l != self.n_layers-1:
                        h = layer(blocks[0], x)
                        if self.aggregator == 'attentional':
                            h= h.flatten(1)
                    else:
                        h = layer(blocks[0], x)
                        if self.aggregator == 'attentional':
                            h = h.mean(1)
                        h = self.encoder.gs_mu(h)
                        if self.supervised:
                            n = blocks[-1].dstdata['ngh']
                            hm,_,_,_ = self.encoder_molecule(n)
                            h = h*hm
                            px_scale, px_r, px_l= self.decoder(h)
                            #px_scale = px_scale*th.exp(h)
                            px_scale = h @ self.encoder_molecule.module2celltype
                            #px_scale = px_scale.softmax(dim=-1)
                            p_class[output_nodes] = px_scale.cpu().detach()

                    y[output_nodes] = h.cpu().detach()#.numpy()
                g.ndata['h'] = y
            return y, p_class

    def inference_attention(self, g, device, batch_size, num_workers, nodes=None,buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        if type(nodes) == type(None):
            nodes = th.arange(g.num_nodes()).to(g.device)

        g.ndata['h'] = th.log(g.ndata['gene']+1)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.NodeDataLoader(
                g, nodes, sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                persistent_workers=(num_workers > 0))

        if buffer_device is None:
            buffer_device = device
        
        for l, layer in enumerate(self.encoder.encoder_dict['GS']):
            if l == self.n_layers - 1:
                    y = th.zeros(g.num_nodes(), self.n_latent,device=buffer_device)
                    att2_list = [] #if not self.supervised else th.zeros(g.num_nodes(), self.n_classes)
            else:
                    y = th.zeros(g.num_nodes(), self.n_hidden*4, device=buffer_device)
                    att1_list = []
                
            for input_nodes, output_nodes, blocks in dataloader:
                x = blocks[0].srcdata['h']
                if l != self.n_layers-1:
                    h,att1 = layer(blocks[0], x,get_attention=True)
                    att1_list.append(att1.mean(1).cpu().detach())
                    h= h.flatten(1)
                    
                else:
                    h, att2 = layer(blocks[0], x,get_attention=True)
                    att2_list.append(att2.mean(1).cpu().detach())
                    h = h.mean(1)
                    h = self.encoder.gs_mu(h)   
                y[output_nodes] = h.cpu().detach().to(buffer_device)
            g.ndata['h'] = y
        return th.concat(att1_list), th.concat(att2_list)

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
            self.norm = F.normalize#PairNorm()#DiffGroupNorm(n_hidden,n_classes,None)
        else:
            self.norm = F.normalize#PairNorm()#DiffGroupNorm(n_hidden,20)
        self.num_heads = 4
        self.n_layers = n_layers
        for i in range(0,n_layers-1):
            if i > 0:
                in_feats = n_hidden
                x = 0.2
            else:
                x = 0

            if aggregator == 'attentional':
                layers.append(dglnn.GATv2Conv(in_feats, 
                                            n_hidden, 
                                            num_heads=self.num_heads,
                                            feat_drop=x,
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
            layers.append(dglnn.GATv2Conv(n_hidden*self.num_heads, 
                                        n_hidden, 
                                        num_heads=self.num_heads, 
                                        feat_drop=dropout,
                                        #allow_zero_in_degree=False
                                        ))

        else:
            layers.append(dglnn.SAGEConv(n_hidden, 
                                            n_hidden, 
                                            aggregator_type=aggregator,
                                            feat_drop=dropout,
                                            activation=F.relu,
                                            norm=self.norm
                                            ))

        self.encoder_dict = nn.ModuleDict({'GS': layers})
        self.gs_mu = nn.Linear(n_hidden, n_latent)
        self.gs_var = nn.Linear(n_hidden, n_latent)
    
    def forward(self, x, blocks=None): 
        h = th.log(x+1)
        for l, (layer, block) in enumerate(zip(self.encoder_dict['GS'], blocks)):
            if self.aggregator != 'attentional':
                h = layer(block, h,)
            else:
                if l != self.n_layers-1:
                    h = layer(block, h,).flatten(1)
                else:
                    h = layer(block, h,).mean(1)

        z_loc = self.gs_mu(h)
        z_scale = th.exp(self.gs_var(h)) +1e-6
        return z_loc, z_scale

class EncoderMolecule(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_latent,

        ):
        super().__init__()

        self.softplus = nn.Softplus()
        self.fc = FCLayers(in_feats, n_hidden)
        self.mu = nn.Linear(n_hidden, n_latent)
        self.var = nn.Linear(n_hidden, n_latent)

        self.fc_l =FCLayers(in_feats, n_hidden)   
        self.mu_l = nn.Linear(n_hidden, 1)
        self.var_l = nn.Linear(n_hidden, 1)

        self.alpha_gene = th.nn.Parameter(th.randn(in_feats))
        self.module2celltype = th.nn.Parameter(th.randn([n_latent ,n_classes]))
        self.dispersion = th.nn.Parameter(th.randn([in_feats]))

    def forward(self,x):
        x = th.log(x+1)   
        h = self.fc(x)
        z_loc = self.mu(h)
        z_scale = th.exp(self.var(h)) +1e-6

        hl = self.fc_l(x)
        l_loc = self.mu_l(hl)
        l_scale = th.exp(self.var_l(hl)) +1e-6
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

        self.fc = FCLayers(n_latent,n_hidden,n_layers=2)

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_classes),
            nn.Softmax(dim=-1)
            )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, in_feats)
        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, 1)

    def forward(self, z):

        px = self.fc(z)

        px_scale = self.px_scale_decoder(px)
        px_l = self.px_dropout_decoder(px)
        px_r = self.px_r_decoder(px)
        return px_scale, px_r, px_l