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
                 lr=1e-4,
                 features_name='gene',
                 supervised=False,
                 reference=0,
                 smooth=False,
                 device='cpu',
                 aggregator='attentional',
                 inference_type='deterministic',
                 loss_type='unsupervised',#or supervised
                 #decoder_loss=True,
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
                dr =  mfgs[-1].dstdata[self.features_name]

            if len(batch_inputs.shape) == 1:
                if self.supervised == False:
                    #batch_inputs = F.one_hot(batch_inputs.to(th.int64), num_classes=self.in_feats)
                    batch_inputs = batch_inputs.to(th.int64)

            zn_loc = self.module.encoder(batch_inputs,mfgs, dr=dr)
            if self.loss_type == 'unsupervised':
                graph_loss = self.loss_fcn(zn_loc, pos, neg).mean()
                '''decoder_n1 = self.module.encoder.decoder(zn_loc).softmax(dim=-1)
                feats_n1 = F.one_hot((mfgs[-1].srcdata[self.features_name]), num_classes=self.in_feats).T
                #feats_n1 = (th.tensor(feats_n1,dtype=th.float32)@adjacency_matrix.to(self.device)).T
                feats_n1 = th.sparse.mm(
                    th.tensor(feats_n1,dtype=th.float32).to_sparse_coo(),
                    mfgs[-1].adjacency_matrix().to(self.device)
                    ).to_dense().T
                feats_n1 = feats_n1.softmax(dim=-1)
                #print(feats_n1.shape, decoder_n1.shape)
                graph_loss += - nn.CosineSimilarity(dim=1, eps=1e-08)(decoder_n1, feats_n1).mean(axis=0)'''

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
                                n_hidden= 64,
                                n_latent= 64,
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
            g.ndata['h'] = g.ndata[self.features_name]#.long()
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])

            dataloader = dgl.dataloading.DataLoader(
                    g, th.arange(g.num_nodes()).to(g.device), sampler, device=device,
                    batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                    persistent_workers=(num_workers > 0))

            for l, layer in enumerate(self.encoder.encoder_dict['GS']):
                if l == self.n_layers - 1:
                    y = th.zeros(g.num_nodes(), self.encoder.n_embed) #if not self.supervised else th.zeros(g.num_nodes(), self.n_classes)
                else:
                    if self.aggregator == 'attentional':
                        y = th.zeros(g.num_nodes(), self.encoder.n_embed*self.encoder.num_heads)

                if self.supervised:
                    p_class = th.zeros(g.num_nodes(), self.n_classes)
                else:
                    p_class = None

                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    x = blocks[0].srcdata['h']
                    if l != self.n_layers-1:
                        h = layer(blocks[0], x)
                        h = h.flatten(1)
                        
                    else:
                        h = layer(blocks[0], x)
                        h = h.mean(1)
                        h = self.encoder.fw(h)

                    y[output_nodes] = h.cpu().detach()#.numpy()
                g.ndata['h'] = y
            return y, p_class

    def inference_attention(self, g, device, batch_size, num_workers, nodes=None, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        if type(nodes) == type(None):
            nodes = th.arange(g.num_nodes()).to(g.device)


        g.ndata['h'] = g.ndata[self.features_name].long()
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.NodeDataLoader(
                g, nodes, sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                persistent_workers=(num_workers > 0))

        if buffer_device is None:
            buffer_device = device
        self.attention_list = [[] for x in range(self.n_layers)]
        
        for l, layer in enumerate(self.encoder.encoder_dict['GS']):
            if l == self.n_layers - 1:
                    y = th.zeros(g.num_nodes(), self.encoder.n_embed, device=buffer_device)
            else:
                    y = th.zeros(g.num_nodes(), self.encoder.n_embed*self.encoder.num_heads, device=buffer_device)
                
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                #x = blocks[0].srcdata['h']
                if l == 0:
                    x = self.encoder.embedding(blocks[0].srcdata['h'])
                else:
                    x = blocks[0].srcdata['h']
                dr = self.encoder.embedding(blocks[0].dstdata[self.features_name])#.long()
                if l != self.n_layers-1:
                    h,att = layer(blocks[0], x,get_attention=True)
                    #att1_list.append(att1.mean(1).cpu().detach())
                    self.attention_list[l].append(att.cpu().detach())
                    h= h.flatten(1)
                    
                else:
                    h, att = layer(blocks[0], x,get_attention=True)
                    #att2_list.append(att2.mean(1).cpu().detach())
                    self.attention_list[l].append(att.cpu().detach())
                    h = h.mean(1)
                    
                    h = self.encoder.ln1(h) + self.encoder.embedding(dr)
                    h = self.encoder.fw(self.encoder.ln2(h)) + h

                y[output_nodes] = h.cpu().detach().to(buffer_device)
            g.ndata['h'] = y
        return [th.concat(a) for a in self.attention_list]

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
        n_embed = 64
        self.n_embed = n_embed
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

        layers = nn.ModuleList()
        self.num_heads = 4
        self.n_layers = n_layers
        for i in range(0,n_layers-1):
            if i == 0:
                layers.append(dglnn.GATv2Conv(in_feats, 
                                            n_hidden, 
                                            num_heads=self.num_heads,
                                            feat_drop=dropout,
                                            residual=True,
                                            #allow_zero_in_degree=False
                                            ))
            else:
                layers.append(dglnn.GATv2Conv(n_embed*self.num_heads, 
                            n_hidden, 
                            num_heads=self.num_heads,
                            feat_drop=dropout,
                            residual=True,
                            #allow_zero_in_degree=False
                            ))


        layers.append(dglnn.GATv2Conv(n_embed*self.num_heads, 
                                    n_latent, 
                                    num_heads=self.num_heads, 
                                    feat_drop=dropout,
                                    residual=True,
                                    #allow_zero_in_degree=False
                                    ))

        self.encoder_dict = nn.ModuleDict({'GS': layers})
        #self.fw = nn.Linear(n_hidden, n_embed)
        self.fw = nn.Sequential(
            nn.Linear(n_latent, 4 * n_latent),
            nn.BatchNorm1d(n_latent * 4, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Linear(4 * n_latent, n_latent),
        )
    
    def forward(self, x, blocks=None, dr=0): 
        h = th.log(x+1)
        for l, (layer, block) in enumerate(zip(self.encoder_dict['GS'], blocks)):
            if self.aggregator != 'attentional':
                h = layer(block, h,)
            else:
                if l != self.n_layers-1:
                    h = layer(block, h,).flatten(1)
                else:
                    h = layer(block, h,).mean(1)
        
        h = self.fw(h)
        return h