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
from torch.distributions import Normal, kl_divergence as kl

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

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

class SAGELightning(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation=F.relu,
                 dropout=0.2,
                 lr=0.001,
                 supervised=False,
                 kappa=0,
                 Ncells=0,
                 reference=0,
                 smooth=False,
                 device='cpu',
                 aggregator='attentional',
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.module = SAGE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, supervised,aggregator)
        self.lr = lr
        self.supervised= supervised
        self.loss_fcn = CrossEntropyLoss()
        self.kappa = kappa
        self.reference=th.tensor(reference,dtype=th.float32)
        self.smooth = smooth
        if self.supervised:
            #self.automatic_optimization = False
            #self.sl = SemanticLoss(n_hidden,n_classes,ncells=Ncells,device=device)
            self.train_acc = torchmetrics.Accuracy()
            p = th.tensor(Ncells*reference.sum(axis=0),dtype=th.float32,device=self.device)
            self.p = p/p.sum()
            self.kl = th.nn.KLDivLoss(reduction='sum')
            

    def training_step(self, batch, batch_idx):
        if self.supervised:
            opt = self.optimizers()
        batch1 = batch['unlabelled']
        _, pos_graph, neg_graph, mfgs = batch1
        mfgs = [mfg.int() for mfg in mfgs]
        #pos_graph = pos_graph.to(self.device)
        #neg_graph = neg_graph.to(self.device)
        batch_inputs_u = mfgs[0].srcdata['gene']
        batch_pred_unlab, qz_m, qz_v = self.module(mfgs, batch_inputs_u)
        bu = batch_inputs_u[pos_graph.nodes()]
        loss,pos, neg = self.loss_fcn(batch_pred_unlab, pos_graph, neg_graph) #* 5
        
        if self.supervised:
            #bu = batch_inputs_u[pos_graph.nodes()]
            mean = th.zeros_like(qz_m)
            scale = th.ones_like(qz_v)

            kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
                dim=1)

            
            pass


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
        '''if self.supervised:
            d_opt = th.optim.Adam(self.module.domain_adaptation.parameters(), lr=1e-3)
            return [optimizer, d_opt]'''
        return optimizer

    def loss_discriminator(self, latent_tensors, 
        predict_true_class: bool = True,
        return_details: bool = False,
        ):
        n_classes = 2
        losses = []
        for i, z in enumerate(latent_tensors):
            cls_logits = self.module.domain_adaptation(z)

            if predict_true_class:
                cls_target = th.zeros_like(cls_logits,dtype=th.float32,device=z.device)
                cls_target[:,i] = 1
            else:
                cls_target = th.ones_like(cls_logits,dtype=th.float32,device=z.device)
                cls_target[:,i] = 0.0

            bcloss = th.nn.BCEWithLogitsLoss()(cls_logits,cls_target)
            losses.append(bcloss)

        if return_details:
            return losses

        total_loss = th.stack(losses).sum()
        return total_loss

class SAGE(nn.Module):
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
            self.domain_adaptation = Classifier(n_input=n_hidden,
                                                n_labels=2,
                                                softmax=False,
                                                reverse_gradients=True)

        self.encoder = Encoder(in_feats,
                                n_hidden,
                                n_classes,
                                n_layers,
                                supervised,
                                aggregator)
        self.mean_encoder = self.encoder.encoder_dict['mean']
        self.var_encoder = self.encoder.encoder_dict['var']

    def forward(self, blocks, x):
        h = th.log(x+1)   
        for l, (layer, block) in enumerate(zip(self.encoder.encoder_dict['GS'], blocks)):
            feat_n = []
            if self.aggregator != 'attentional':
                h = layer(block, h,)#.mean(1)
                #h = self.encoder.encoder_dict['FC'][l](h)
            else:
                h = layer(block, h,).mean(1)

                #h = self.encoder.encoder_dict['FC'][l](h)
        
        mu,var = self.mean_encoder(h), self.var_encoder(h)
        h = reparameterize_gaussian(mu,th.exp(var)+1e-8)
        #h = self.encoder.encoder_dict['FC'][1](h)
        return h,mu,var

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
                y = th.zeros(g.num_nodes(), self.n_hidden) #if not self.supervised else th.zeros(g.num_nodes(), self.n_classes)
            else: 
                y = th.zeros(g.num_nodes(), self.n_hidden)

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
                block = blocks[0]
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

                if layer == 1:
                    h = self.mean_encoder(h)
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
        

            self.mean_encoder = nn.Linear(n_hidden, n_hidden)
            self.var_encoder = nn.Linear(n_hidden, n_hidden)
            layers = nn.ModuleList()

            if supervised:
                self.norm = PairNorm()
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

            if supervised:
                classifier = Classifier(n_input=n_hidden,
                                        n_labels=n_classes,
                                        softmax=False,
                                        reverse_gradients=False)
            else:
                classifier = None

            self.encoder_dict = nn.ModuleDict({'GS': layers, 
                                                'mean': self.mean_encoder,
                                                'var':self.var_encoder,
                                                'CF':classifier})