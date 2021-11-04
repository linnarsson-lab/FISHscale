from matplotlib.pyplot import bone
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
from torch.autograd import Function

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
                 lr=0.001,
                 supervised=False,
                 kappa=0,
                 Ncells=0,
                 reference=0,
                 device='cpu'
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.module = SAGE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, supervised)
        self.lr = lr
        self.supervised= supervised
        self.loss_fcn = CrossEntropyLoss()
        self.kappa = kappa
        self.reference=th.tensor(reference,dtype=th.float32)
        if self.supervised:
            self.automatic_optimization = False
            #self.sl = SemanticLoss(n_hidden,n_classes,ncells=Ncells,device=device)
            self.train_acc = torchmetrics.Accuracy()
            p = th.tensor(Ncells*reference.sum(axis=0),dtype=th.float32,device=self.device)
            self.p = p/p.sum()
            self.kl = th.nn.KLDivLoss(reduction='sum')
            

    def training_step(self, batch, batch_idx):
        if self.supervised:
            opt, d_opt = self.optimizers()
        batch1 = batch['unlabelled']
        _, pos_graph, neg_graph, mfgs = batch1
        mfgs = [mfg.int() for mfg in mfgs]
        #pos_graph = pos_graph.to(self.device)
        #neg_graph = neg_graph.to(self.device)
        batch_inputs_u = mfgs[0].srcdata['gene']
        batch_pred_unlab = self.module(mfgs, batch_inputs_u)
        bu = batch_inputs_u[pos_graph.nodes()]
        loss,pos, neg = self.loss_fcn(batch_pred_unlab, pos_graph, neg_graph) #* 5
        
        if self.supervised:
            batch2 = batch['labelled']
            _, pos_graph, neg_graph, mfgs = batch2
            mfgs = [mfg.int() for mfg in mfgs]
            #pos_graph = pos_graph.to(self.device)
            #neg_graph = neg_graph.to(self.device)
            batch_inputs = mfgs[0].srcdata['gene']
            batch_labels = mfgs[-1].dstdata['label']
            bl = batch_inputs[pos_graph.nodes()]
            batch_pred_lab = self.module(mfgs, batch_inputs)
            supervised_loss,_,_ = self.loss_fcn(batch_pred_lab, pos_graph, neg_graph)

            # Label prediction loss
            labels_pred = self.module.encoder.encoder_dict['CF'](batch_pred_lab)
            probabilities_lab = F.softmax(labels_pred,dim=-1)
            cce = th.nn.CrossEntropyLoss()
            classifier_loss = cce(labels_pred,batch_labels) #* 0.05
            self.train_acc(labels_pred.argsort(axis=-1)[:,-1],batch_labels)
            self.log('Classifier Loss',classifier_loss)
            self.log('train_acc', self.train_acc, prog_bar=True, on_step=True)
            
            #Domain Adaptation Loss
            classifier_domain_loss = self.loss_discriminator([batch_pred_unlab, batch_pred_lab],predict_true_class=True)
            self.log('Classifier_true', classifier_domain_loss, prog_bar=False, on_step=True)

            #Semantic Loss
            probabilities_unlab = F.softmax(self.module.encoder.encoder_dict['CF'](batch_pred_unlab),dim=-1)
            labels_unlab = probabilities_unlab.argsort(axis=-1)[:,-1]
            '''self.sl.semantic_loss(pseudo_latent=batch_pred_unlab, 
                        pseudo_labels=labels_unlab ,
                        true_latent=batch_pred_lab,
                        true_labels=labels_pred.argsort(axis=-1)[:,-1],
                        )'''

            # Bonefight regularization of cell types
            bone_fight_loss = -F.cosine_similarity(probabilities_unlab @ self.reference.T.to(self.device), bu,dim=0).mean()
            bone_fight_loss += -F.cosine_similarity(probabilities_unlab @ self.reference.T.to(self.device), bu,dim=1).mean()
            '''q = th.ones(probabilities_unlab.shape[0])/probabilities_unlab.shape[0]
            p = th.log(self.p.T @ probabilities_unlab.T)
            kl_loss = self.kl(p,q)
            bone_fight_loss = bone_fight_loss0 + kl_loss'''

            # Will increasingly apply supervised loss, domain adaptation loss
            # from 0 to 1, from iteration 0 to 200, focusing first on unsupervised 
            # graphsage task
            kappa = 2/(1+10**(-1*((1*self.kappa)/2000)))-1
            self.kappa += 1
            loss = loss*kappa
            loss = classifier_loss + loss + bone_fight_loss + kappa*(kappa*classifier_domain_loss + kappa*supervised_loss) #+ semantic_loss.detach()
            
            opt.zero_grad()
            self.manual_backward(loss,retain_graph=True)
            opt.step()

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
                supervised):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.supervised = supervised
        
        if self.supervised:
            self.domain_adaptation = Classifier(n_input=n_hidden,
                                                n_labels=2,
                                                softmax=False,
                                                reverse_gradients=True)

        self.encoder = Encoder(in_feats,n_hidden,n_classes,n_layers,supervised)
 
    def forward(self, blocks, x):
        h = th.log(x+1)   
        for l, (layer, block) in enumerate(zip(self.encoder.encoder_dict['GS'], blocks)):
            #print(l)
            h = layer(block, h,)
            #h = F.normalize(h)
            h = self.encoder.encoder_dict['FC'][l](h)

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
                h = th.log(x[input_nodes]+1)#.to(device)

                h = layer(block, h)
                h = self.encoder.encoder_dict['FC'][l](h)
                #h = F.normalize(h)
                y[output_nodes] = h.cpu().detach()#.numpy()
            x = y
        return y

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
        reverse_gradients:bool=False,
    ):
        super().__init__()
        self.grl = GradientReversal()
        self.reverse_gradients = reverse_gradients
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
        if self.reverse_gradients:
            x = self.grl(x)
        return self.classifier(x)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(th.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

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

            hidden = [nn.Sequential(
                                nn.Linear(n_hidden , n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU()) for x in range(n_layers )]

            latent = nn.Sequential(
                        nn.Linear(n_hidden , n_hidden), #if not supervised else nn.Linear(n_hidden , self.n_classes),
                        nn.BatchNorm1d(n_hidden), #if not supervised else  nn.BatchNorm1d(self.n_classes),
                        #nn.Softmax()
                        )
            layers = nn.ModuleList()
            if n_layers > 1:
                layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'pool',feat_drop=0.2,activation=F.relu,norm=F.normalize))
                for i in range(1,n_layers):
                    layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'pool',feat_drop=0.2,activation=F.relu,norm=F.normalize))
            else:
                layers.append(dglnn.SAGEConv(in_feats, n_classes, 'pool',feat_drop=0.2,activation=F.relu,norm=F.normalize))

            if supervised:
                classifier = Classifier(n_input=n_hidden,
                                        n_labels=n_classes,
                                        softmax=False,
                                        reverse_gradients=False)
            else:
                classifier = None

            self.encoder_dict = nn.ModuleDict({'GS': layers, 
                                                'BN':bns,
                                                'FC': nn.ModuleList([h for h in hidden]+[latent]),
                                                'CF':classifier})


class SemanticLoss:
    def __init__(self , 
        n_hidden,
        n_labels,
        ncells=0,
        device='cpu'
        ):
        self.dev = device
        self.centroids_pseudo = th.zeros([n_hidden,n_labels],device=self.dev)
        self.pseudo_count = th.ones([n_labels],device=self.dev)
        self.centroids_true = th.zeros([n_hidden, n_labels],device=self.dev)
        self.true_count = th.ones([n_labels],device=self.dev)
        
        '''if type(ncells) == type(0):
            self.ncells = self.true_count/self.true_count.sum()
            self.ncells_max = self.true_count.sum()*1000
        else:
            self.ncells_max = ncells.sum()
            self.ncells = th.tensor(ncells/ncells.sum(),device=self.dev)'''

        super().__init__()
    def semantic_loss(self, 
            pseudo_latent, 
            pseudo_labels, 
            true_latent, 
            true_labels):

        '''if self.true_count.max() >= self.ncells_max/10:
            self.pseudo_count = th.ones([self.pseudo_count.shape[0]],device=self.dev)
            self.true_count = th.ones([self.true_count.shape[0]],device=self.dev)'''

        '''for pl in pseudo_labels.unique():
            filt = pseudo_labels == pl
            if filt.sum() > 10:
                centroid_pl = pseudo_latent[filt,:]
                dp = th.tensor([nn.MSELoss()(centroid_pl[cell,:], self.centroids_pseudo[:,pl]) for cell in range(centroid_pl.shape[0])])
                dispersion_p = th.mean(dp)
                centroid_pl = centroid_pl.mean(axis=0)
                new_avg_pl = self.centroids_pseudo[:,pl] * self.pseudo_count[pl] + centroid_pl *filt.sum()
                new_avg_pl = new_avg_pl/(self.pseudo_count[pl] +filt.sum())
                self.pseudo_count[pl] += filt.sum()
                self.centroids_pseudo[:,pl] = new_avg_pl'''

        for tl in true_labels.unique():
            filt = true_labels == tl
            if filt.sum() > 10:
                centroid_tl = true_latent[filt,:]
                '''dispersion_t = th.mean(th.tensor([nn.MSELoss()(centroid_tl[cell,:], self.centroids_true[:,tl]) for cell in range(centroid_tl.shape[0])],device='cuda'))'''
                centroid_tl = centroid_tl.mean(axis=0)
                new_avg_tl = self.centroids_true[:,tl]* self.true_count[tl] + centroid_tl*filt.sum()
                new_avg_tl = new_avg_tl/(self.true_count[tl] +filt.sum())
                self.true_count[tl] += filt.sum()
                self.centroids_true[:,tl] = new_avg_tl
        
        #kl_density = th.nn.functional.kl_div(self.ncells.log(),self.pseudo_count/self.pseudo_count.sum())
        #kl_density =  -F.logsigmoid((self.ncells*self.pseudo_count).sum(-1)).sum()*100
        #semantic_loss = -F.logsigmoid((self.centroids_pseudo*self.centroids_true).sum(-1)).mean() + kl_density #
        #semantic_loss = nn.MSELoss()(self.centroids_pseudo, self.centroids_true) + kl_density + dispersion_p
        #return semantic_loss