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
                 features_name='gene',
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
                 loss_type='unsupervised',#or supervised
                 ):
        super().__init__()

        self.module = SAGE(in_feats=in_feats, 
                            n_hidden=n_hidden,
                            n_latent=n_latent, 
                            n_classes=n_classes, 
                            n_layers=n_layers,
                            dropout=dropout, 
                            supervised=supervised,
                            aggregator= aggregator,
                            features_name=features_name,
                            )

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
        self.loss_type = loss_type
        self.features_name = features_name

        if self.inference_type == 'VI':
            self.automatic_optimization = False
            self.svi = PyroOptWrap(model=self.model,
                    guide=self.guide,
                    optim=pyro.optim.Adam({"lr": self.lr}),
                    loss=pyro.infer.Trace_ELBO())

        self.automatic_optimization = False
        if self.supervised:
            self.num_classes = n_classes
            '''   
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
            '''

    def training_step(self, batch, batch_idx):

        self.reference = self.reference.to(self.device)
        losses = []
        for sub_batch in batch:
            if self.supervised:
                _, _, mfgs = sub_batch
                mfgs = [mfg.int() for mfg in mfgs]
                batch_inputs = mfgs[0].srcdata[self.features_name]
            
            else:
                _, pos, neg, mfgs = sub_batch
                pos_ids = pos.edges()[0]
                mfgs = [mfg.int() for mfg in mfgs]
                batch_inputs = mfgs[0].srcdata[self.features_name]

            if len(batch_inputs.shape) == 1:
                if self.supervised == False:
                    batch_inputs = F.one_hot(batch_inputs.to(th.int64), num_classes=self.in_feats)

            zn_loc = self.module.encoder(batch_inputs,mfgs)
            if self.loss_type == 'unsupervised':
                graph_loss = self.loss_fcn(zn_loc, pos, neg).mean()
            else:
                graph_loss = F.cross_entropy(zn_loc, mfgs[-1].dstdata['label'])

            opt_g = self.optimizers()
            opt_g.zero_grad()
            self.manual_backward(graph_loss)
            opt_g.step()

            loss = graph_loss
            losses.append(loss)
        loss = th.stack(losses).mean()
        self.log('train_loss', 
            loss, prog_bar=True, on_step=True, on_epoch=True,batch_size=zn_loc.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer_graph = th.optim.Adam(self.module.encoder.parameters(), lr=self.lr)
        #optimizer_nb = th.optim.Adam(self.module.encoder_molecule.parameters(), lr=0.01)
        lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer_graph,)
        scheduler = {
            'scheduler': lr_scheduler, 
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'train_loss'
        }
        return [optimizer_graph],[scheduler]

    def validation_step(self,batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, th.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["train_loss"])

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
                aggregator,
                features_name='gene'):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_classes = n_classes
        self.supervised = supervised
        self.aggregator = aggregator
        self.in_feats = in_feats
        self.features_name = features_name
        self.encoder = Encoder(in_feats=in_feats,
                                n_hidden=128,
                                n_latent=128,
                                n_layers=n_layers,
                                supervised=supervised,
                                aggregator=aggregator,
                                dropout= dropout,
                                )

    def inference(self, g, device, batch_size, num_workers, core_nodes=None):
            """
            Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
            g : the entire graph.
            The inference code is written in a fashion that it could handle any number of nodes and
            layers.
            """
            self.eval()
            if len(g.ndata[self.features_name].shape) == 1:
                g.ndata['h'] = self.encoder.embedding(F.one_hot(g.ndata[self.features_name], num_classes=self.in_feats))
                #g.ndata['h'] = th.log(F.one_hot(g.ndata[self.features_name], num_classes=self.in_feats)+1)
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
                
                if core_nodes is None:
                    dataloader = dgl.dataloading.NodeDataLoader(
                            g, th.arange(g.num_nodes()).to(g.device), sampler, device=device,
                            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                            persistent_workers=(num_workers > 0))
                else:
                    dataloader = dgl.dataloading.NodeDataLoader(
                        g, core_nodes.to(g.device), sampler, device=device,
                        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                        persistent_workers=(num_workers > 0))

            else:
                #g.ndata['h'] = th.log(g.ndata[self.features_name]+1)
                g.ndata['h'] = self.encoder.embedding(g.ndata[self.features_name])
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
                        #h = self.encoder.gs_mu(h)
                    y[output_nodes] = h.cpu().detach()#.numpy()
                g.ndata['h'] = y
            return y, p_class

    def inference_attention(self, g, device, batch_size, num_workers, nodes=None, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        if type(nodes) == type(None):
            nodes = th.arange(g.num_nodes()).to(g.device)

        if len(g.ndata[self.features_name].shape) == 1:
            #g.ndata['h'] = th.log(F.one_hot(g.ndata[self.features_name], num_classes=self.in_feats)+1)
            g.ndata['h'] = self.encoder.embedding(F.one_hot(g.ndata[self.features_name], num_classes=self.in_feats))
        else:
            #g.ndata['h'] = th.log(g.ndata[self.features_name]+1)
            g.ndata['h'] = self.encoder.embedding(g.ndata[self.features_name])
            
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
                    #h = self.encoder.gs_mu(h)   
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
        n_embed = 128
        self.embedding = nn.Embedding(in_feats, n_embed)

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
                                        n_latent, 
                                        num_heads=self.num_heads, 
                                        feat_drop=dropout,
                                        #allow_zero_in_degree=False
                                        ))

        else:
            layers.append(dglnn.SAGEConv(n_hidden, 
                                            n_latent, 
                                            aggregator_type=aggregator,
                                            feat_drop=dropout,
                                            activation=F.relu,
                                            norm=self.norm
                                            ))

        self.encoder_dict = nn.ModuleDict({'GS': layers})
        #self.gs_mu = nn.Linear(n_hidden, n_latent)
        #self.gs_var = nn.Linear(n_hidden, n_latent)
    
    def forward(self, x, blocks=None): 
        #h = th.log(x+1)
        e = self.embedding(x)
        h = e
        for l, (layer, block) in enumerate(zip(self.encoder_dict['GS'], blocks)):
            if self.aggregator != 'attentional':
                h = layer(block, h,)
            else:
                if l != self.n_layers-1:
                    h = layer(block, h,).flatten(1)
                else:
                    h = layer(block, h,).mean(1)
            h = h + e

        #z_loc = self.gs_mu(h)
        #z_scale = th.exp(self.gs_var(h)) +1e-6
        return h