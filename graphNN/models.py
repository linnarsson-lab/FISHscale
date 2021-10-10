import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import accuracy
import torchmetrics
import math
import numpy as np
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import tqdm
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

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

class SAGELightning(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation=F.relu,
                 dropout=0.2,
                 lr=0.01,
                 supervised=False,
                 kappa=0,
                 Ncells=0
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.module = SAGE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, supervised)
        self.lr = lr
        self.supervised= supervised
        self.loss_fcn = CrossEntropyLoss()
        self.kappa = kappa
        #self.automatic_optimization = True
        if self.supervised:
            self.automatic_optimization = False
            self.sl = SemanticLoss(n_hidden,n_classes,device=self.device,ncells=Ncells)
            self.train_acc = torchmetrics.Accuracy()
            

    def training_step(self, batch, batch_idx):
        if self.supervised:
            opt, d_opt = self.optimizers()
        batch1 = batch['unlabelled']
        _, pos_graph, neg_graph, mfgs = batch1
        mfgs = [mfg.int() for mfg in mfgs]
        #pos_graph = pos_graph.to(self.device)
        #neg_graph = neg_graph.to(self.device)
        batch_inputs = mfgs[0].srcdata['gene']
        batch_pred_unlab = self.module(mfgs, batch_inputs)
        loss,pos, neg = self.loss_fcn(batch_pred_unlab, pos_graph, neg_graph) #* 5
        

        if self.supervised:
            batch2 = batch['labelled']
            _, pos_graph, neg_graph, mfgs = batch2
            mfgs = [mfg.int() for mfg in mfgs]
            #pos_graph = pos_graph.to(self.device)
            #neg_graph = neg_graph.to(self.device)
            batch_inputs = mfgs[0].srcdata['gene']
            batch_labels = mfgs[-1].dstdata['label']
            batch_pred_lab = self.module(mfgs, batch_inputs)
            supervised_loss,_,_ = self.loss_fcn(batch_pred_lab, pos_graph, neg_graph)

            # Label prediction loss
            labels_pred = self.module.encoder.encoder_dict['CF'](batch_pred_lab)
            cce = th.nn.CrossEntropyLoss()
            classifier_loss = cce(labels_pred,batch_labels) #* 0.05
            self.train_acc(labels_pred.argsort(axis=-1)[:,-1],batch_labels)
            self.log('Classifier Loss',classifier_loss)
            self.log('train_acc', self.train_acc, prog_bar=True, on_step=True)

            #Domain Adaptation Loss
            classifier_domain_loss = 10*self.loss_discriminator([batch_pred_unlab.detach(), batch_pred_lab.detach()],predict_true_class=True)
            self.log('Classifier_true', classifier_domain_loss, prog_bar=False, on_step=True)
            d_opt.zero_grad()
            self.manual_backward(classifier_domain_loss)
            d_opt.step()

            domain_loss_fake = 10*self.loss_discriminator([batch_pred_unlab, batch_pred_lab],predict_true_class=False)
            self.log('Classifier_fake', domain_loss_fake, prog_bar=False, on_step=True)

            #Semantic Loss
            labels_unlab = self.module.encoder.encoder_dict['CF'](batch_pred_unlab).argsort(axis=-1)[:,-1]
            '''semantic_loss = self.sl.semantic_loss(pseudo_latent=batch_pred_unlab, 
                                                    pseudo_labels=labels_unlab ,
                                                    true_latent=batch_pred_lab,
                                                    true_labels=labels_pred.argsort(axis=-1)[:,-1],
                                                    )
            self.log('Semantic_loss', semantic_loss, prog_bar=True, on_step=True)'''

            
            # Will increasingly apply supervised loss, domain adaptation loss
            # from 0 to 1, from iteration 0 to 200, focusing first on unsupervised 
            # graphsage task
            #kappa = 2/(1+10**(-1*((1*self.kappa)/200)))-1
            #self.kappa += 1
            loss = loss*0.005
            loss += domain_loss_fake + supervised_loss*0.005 + classifier_loss #+ semantic_loss.detach()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        if self.supervised == False:
            self.log('balance', pos/neg, prog_bar=True,on_step=True, on_epoch=True,)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int() for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['gene']
        batch_pred = self.module(mfgs, batch_inputs)
        return batch_pred

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.module.encoder.parameters(), lr=self.lr)
        if self.supervised:
            d_opt = th.optim.Adam(self.module.domain_adaptation.parameters(), lr=1e-3)
            return [optimizer, d_opt]
        else:
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
                cls_target = th.zeros(
                    n_classes, dtype=th.float32, device=z.device
                )
                cls_target[i] = 1.0
            else:
                cls_target = th.ones(
                    n_classes, dtype=th.float32, device=z.device
                ) / (n_classes - 1)
                cls_target[i] = 0.0

            l_soft = cls_logits * cls_target
            cls_loss = -l_soft.sum(dim=1).mean()
            losses.append(cls_loss)

        if return_details:
            return losses

        total_loss = th.stack(losses).sum()
        return total_loss

class SemanticLoss(nn.Module):
    def __init__(self , 
        n_hidden,
        n_labels,
        device,
        ncells=0,
        ):
        self.device = 'cpu'
        self.centroids_pseudo = th.randn([n_hidden,n_labels],device=self.device)
        self.pseudo_count = th.ones([n_labels],device=self.device)
        self.centroids_true = th.randn([n_hidden, n_labels],device=self.device)
        self.true_count = th.ones([n_labels],device=self.device)
        
        if type(ncells) == type(0):
            self.ncells = self.true_count/self.true_count.sum()
            self.ncells_max = self.true_count.sum()*1000
        else:
            self.ncells_max = ncells.sum()
            self.ncells = th.tensor(ncells/ncells.sum(),device=self.device)

        super().__init__()
    def semantic_loss(self, 
            pseudo_latent, 
            pseudo_labels, 
            true_latent, 
            true_labels):

        if self.pseudo_count.max() >= self.ncells_max:
            self.pseudo_count = th.ones([self.pseudo_count.shape[0]],device=self.device)

        for pl in pseudo_labels.unique():
            filt = pseudo_labels == pl
            if filt.sum() > 5:
                centroid_pl = pseudo_latent[filt,:]
                dp = th.tensor([nn.MSELoss()(centroid_pl[cell,:], self.centroids_pseudo[:,pl]) for cell in range(centroid_pl.shape[0])],device=self.device)
                dispersion_p = th.mean(dp)
                centroid_pl = centroid_pl.mean(axis=0)
                new_avg_pl = self.centroids_pseudo[:,pl] * self.pseudo_count[pl] + centroid_pl *filt.sum()
                new_avg_pl = new_avg_pl/(self.pseudo_count[pl] +filt.sum())
                self.pseudo_count[pl] += filt.sum()
                self.centroids_pseudo[:,pl] = new_avg_pl

        for tl in true_labels.unique():
            filt = true_labels == tl
            if filt.sum() > 5:
                centroid_tl = true_latent[filt,:]
                '''dispersion_t = th.mean(th.tensor([nn.MSELoss()(centroid_tl[cell,:], self.centroids_true[:,tl]) for cell in range(centroid_tl.shape[0])],device='cuda'))'''
                centroid_tl = centroid_tl.mean(axis=0)
                new_avg_tl = self.centroids_true[:,tl]* self.true_count[tl] + centroid_tl*filt.sum()
                new_avg_tl = new_avg_tl/(self.true_count[tl] +filt.sum())
                self.true_count[tl] += filt.sum()
                self.centroids_true[:,tl] = new_avg_tl
        
        kl_density = th.nn.functional.kl_div(self.ncells.log(),self.pseudo_count/self.pseudo_count.sum())
        #kl_density =  -F.logsigmoid((self.ncells*self.pseudo_count).sum(-1)).sum()*100
        #semantic_loss = -F.logsigmoid((self.centroids_pseudo*self.centroids_true).sum(-1)).mean() + kl_density #+ dispersion_p
        semantic_loss = nn.MSELoss()(self.centroids_pseudo, self.centroids_true) + kl_density
        return semantic_loss

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
                supervised):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.supervised = supervised
        
        if self.supervised:
            self.domain_adaptation = Classifier(n_input=n_hidden,n_labels=2,softmax=False)

        self.encoder = Encoder(in_feats,n_hidden,n_classes,n_layers,supervised)
        
    def forward(self, blocks, x):
        h = x   
        for l, (layer, block) in enumerate(zip(self.encoder.encoder_dict['GS'], blocks)):
            #print(l)
            h = layer(block, h)
            h = F.normalize(h)

            if l != len(self.encoder.encoder_dict['GS']) - 1: #and l != len(self.layers) - 2:
                h = self.encoder.encoder_dict['BN'][l](h)
                h = h.relu()
                h = F.dropout(h, p=0.2, training=self.training)
                h = self.encoder.encoder_dict['FC'][0](h)
        h = self.encoder.encoder_dict['FC'][1](h)
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
        for l, layer in enumerate(self.encoder.encoder_dict['GS']):
            y = th.zeros(g.num_nodes(), self.n_hidden) #if not self.supervised else th.zeros(g.num_nodes(), self.n_classes)
            print(y.shape)

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
                block = block.int()#.to(device)
                h = x[input_nodes]#.to(device)
                h = layer(block, h)
                h = F.normalize(h)

                if l != len(self.encoder.encoder_dict['GS']) -1:# and l != len(self.layers) - 2:
                    h = self.encoder.encoder_dict['BN'][l](h)
                    h = h.relu()
                    h = F.dropout(h, p=0.2, training=self.training)
                    h =self.encoder.encoder_dict['FC'][0](h)
                elif l == len(self.encoder.encoder_dict['GS']) -1:
                    h = self.encoder.encoder_dict['FC'][1](h)

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
            ):
            super().__init__()
        
            bns = nn.ModuleList()
            for _ in range(n_layers):
                bns.append(nn.BatchNorm1d(n_hidden))

            hidden = nn.Sequential(
                                nn.Linear(n_hidden , n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU())

            latent = nn.Sequential(
                        nn.Linear(n_hidden , n_hidden), #if not supervised else nn.Linear(n_hidden , self.n_classes),
                        nn.BatchNorm1d(n_hidden), #if not supervised else  nn.BatchNorm1d(self.n_classes),
                        #nn.Softmax()
                        )
            layers = nn.ModuleList()
            if n_layers > 1:
                layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'pool'))
                for i in range(1,n_layers):
                    layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'pool'))
            else:
                layers.append(dglnn.SAGEConv(in_feats, n_classes, 'pool'))

            if supervised:
                classifier = Classifier(n_input=n_hidden,n_labels=n_classes,softmax=False)
            else:
                classifier = None

            self.encoder_dict = nn.ModuleDict({'GS': layers, 
                                                'BN':bns,
                                                'FC': nn.ModuleList([hidden,latent]),
                                                'CF':classifier})

class Classifier(nn.Module):
    """Basic fully-connected NN classifier
    """

    def __init__(
        self,
        n_input,
        n_hidden=24,
        n_labels=5,
        n_layers=1,
        dropout_rate=0.1,
        softmax=False,
        use_batch_norm: bool=True,
        bias: bool=True,
        use_relu:bool=True,
    ):
        super().__init__()
        layers = [nn.Sequential(
                            nn.Linear(n_input , n_hidden, bias=bias),
                            nn.BatchNorm1d(n_hidden) if use_batch_norm else None,
                            nn.ReLU() if use_relu else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None),
            nn.Linear(n_hidden, n_labels),]

        if softmax:
            layers.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return F.log_softmax(self.classifier(x),dim=-1)


