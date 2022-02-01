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

class SAGELightning(PyroModule):
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
                 ):
        super().__init__()

        self.module = SAGE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, supervised,aggregator)
        self.lr = lr
        self.supervised= supervised
        self.loss_fcn = CrossEntropyLoss()
        self.kappa = 0
        self.reference=th.tensor(reference,dtype=th.float32)
        self.smooth = smooth
        if self.supervised:
            automatic_optimization = False
            self.train_acc = torchmetrics.Accuracy()
            self.kl = th.nn.KLDivLoss(reduction='sum')
            self.dist = celltype_distribution
            self.ncells = ncells

    def forward(self, x):
        _, n, mfgs = x
        x = mfgs[1].dstdata['ngh']
        mfgs = [mfg.int() for mfg in mfgs]
        batch_inputs_u = mfgs[0].srcdata['gene']
        # register PyTorch module `decoder` with Pyro
        embedding = self.module(mfgs, batch_inputs_u)

        hyp_alpha, hyp_beta = th.tensor(9.0),th.tensor(3.0)
        alpha_g_phi_hyp = pyro.sample("alpha_g_phi_hyp",
                dist.Gamma(hyp_alpha, hyp_beta),
        )
                
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([1, x.shape[1]]).to_event(1),
        )  # (self.n_batch, self.n_vars)

        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            c_s = pyro.sample('scale',
                dist.Gamma(th.tensor(3.0),th.tensor(9.0)))
            probabilities_unlab = F.softmax(embedding, dim=-1)
            #print(probabilities_unlab.shape)
            #print(self.reference.T.shape)
            mu = (probabilities_unlab.T*c_s).T @ self.reference.T 
            mu = mu
            alpha = th.ones_like
            alpha = 1/th.tensor(alpha_g_inverse).pow(2)
            rate = alpha/mu

            # score against actual images
            pyro.sample("obs", dist.GammaPoisson(concentration=alpha, rate=rate).to_event(1), obs=x)

            
    def training_step(self, batch, batch_idx):
        batch1 = batch#['unlabelled']
        _, pos_graph, neg_graph, mfgs = batch1
        mfgs = [mfg.int() for mfg in mfgs]
        batch_inputs_u = mfgs[0].srcdata['gene']
        embedding = self.module(mfgs, batch_inputs_u)
        bu = batch_inputs_u[pos_graph.nodes()]
        
        if self.supervised:
            #class_ = self.module.encoder.encoder_dict['CF'](embedding)
            probabilities_unlab = F.softmax(embedding, dim=-1)
            graph_loss,pos, neg = self.loss_fcn(embedding, pos_graph, neg_graph)
            local_nghs = mfgs[0].srcdata['ngh'][pos_graph.nodes()]

            input_local_ngh = th.log(local_nghs+1)
            alpha = 1/th.exp(self.module.encoder.a_d(input_local_ngh)).mean(axis=0)#.pow(2)
            #m_g = th.exp(self.module.encoder.m_g(local_nghs)+1e-6)
            y_s = F.softplus(self.module.encoder.y_s(input_local_ngh))
            c_s = F.softplus(self.module.encoder.c_s(input_local_ngh))

            mu = (probabilities_unlab) @ self.reference.T 
            mu = mu * y_s
            rate = alpha/mu

            NB = GammaPoisson(concentration=alpha,rate=rate)#.log_prob(local_nghs).mean(axis=-1).mean()
            nb_loss = -NB.log_prob(local_nghs).mean(axis=-1).mean()
        
            # Regularize by local nodes
            # Add Predicted same class nodes together.
            if type(self.dist) != type(None):
                #option 2
                p = th.ones(probabilities_unlab.shape[0]) @ probabilities_unlab
                p = th.log(p/p.sum())
                loss_dist = self.kl(p,self.dist.to(self.device)).sum()

                '''p = local_nghs.sum(axis=1) @ probabilities_unlab
                #p = th.ones(probabilities_unlab.shape[0]) @ probabilities_unlab
                #option 1 
                loss_dist = []
                sum_nghs = local_nghs.sum(axis=0)
                for n in range(p.shape[0]):
                    c = int((probabilities_unlab.argsort(dim=-1)[:,-1] == n).sum()) + 1
                    #print(c)
                    m_cell = Multinomial(c, probs=self.dist.to(self.device))

                    loss_dist.append(- Multinomial(total_count=c, probs=p/p.sum()).log_prob(m_cell.sample())/c)
                loss_dist = th.stack(loss_dist).mean()'''
            else:
                loss_dist = 0

            loss = graph_loss + nb_loss + loss_dist
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('Loss Dist', loss_dist, prog_bar=True, on_step=True, on_epoch=True)
            self.log('nb_loss', nb_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('Graph Loss', graph_loss, prog_bar=False, on_step=True, on_epoch=False)

        else:
            graph_loss,pos, neg = self.loss_fcn(embedding, pos_graph, neg_graph)
            loss = graph_loss
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int() for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['gene']
        batch_pred = self.module(mfgs, batch_inputs)
        return batch_pred

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.lr)
        return optimizer


class SAGE(PyroModule):
    '''def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)'''

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
        if self.supervised:
            n_hidden =n_classes

        self.encoder = Encoder(in_feats,
                                n_hidden,
                                n_classes,
                                n_layers,
                                supervised,
                                aggregator)

        print('nclasses',n_hidden)
        self.encoder_latent = PyroModule[nn.Linear](n_hidden,n_hidden)
        self.encoder_latent.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden,n_hidden]).to_event(2))
        self.encoder_latent.bias = PyroSample(dist.Normal(0., 10.).expand([n_hidden]).to_event(1))

    def forward(self, blocks, x):
        h = th.log(x+1)   
        for l, (layer, block) in enumerate(zip(self.encoder.encoder_dict['GS'], blocks)):
            if self.aggregator != 'attentional':
                h = layer(block, h,)#.mean(1)
                #h = self.encoder.encoder_dict['FC'][l](h)
            else:
                h = layer(block, h,).mean(1)
        
        h = self.encoder_latent(h)
        #h = self.encoder.encoder_dict['FC'][l](h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
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
            if l ==  0:
                y = th.zeros(g.num_nodes(), self.n_classes) #if not self.supervised else th.zeros(g.num_nodes(), self.n_classes)
            else: 
                y = th.zeros(g.num_nodes(), self.n_classes)

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

                '''if l == 1:
                    h = self.encoder_latent(h)'''

                #    h = self.mean_encoder(h)#, th.exp(self.var_encoder(h))+1e-4 )
                y[output_nodes] = h.cpu().detach()#.numpy()
            x = y
    
        return y

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
        
            layers = nn.ModuleList()

            if supervised:
                self.norm = F.normalize#DiffGroupNorm(n_hidden,n_classes,None) 
                n_hidden = n_classes
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

            self.encoder_dict = nn.ModuleDict({'GS': layers})

            # Adjust for each ngh
            self.m_g = nn.Linear(in_feats, in_feats)
            # Adjust for batch effects, currently makes no sense, should have
            # shape (batch_size, n_batches), but only 1 batch is implemented now.
            self.y_s = nn.Linear(in_feats, 1)

            #Cell types_per_location
            self.c_s = nn.Linear(in_feats,n_classes)
            # Adjust for technology effects
            #self.t_ngh = nn.Linear(1,1)
            # Adjust overdispersion parameter. This seems to help.
            self.a_d = nn.Linear(in_feats,in_feats)