import re
from threading import local
from dgl.convert import graph
from numpy.random import poisson
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
from torch.distributions import Gamma,Normal, NegativeBinomial,Multinomial, kl_divergence as kl

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
            self.ncells = Ncells

    def training_step(self, batch, batch_idx):
        batch1 = batch#['unlabelled']
        self.reference = self.reference.to(self.device)
        _, pos_graph, neg_graph, mfgs = batch1
        mfgs = [mfg.int() for mfg in mfgs]
        batch_inputs_u = mfgs[0].srcdata['gene']
        batch_pred_unlab = self.module(mfgs, batch_inputs_u)
        bu = batch_inputs_u[pos_graph.nodes()]
        graph_loss,pos, neg = self.loss_fcn(batch_pred_unlab, pos_graph, neg_graph)
        
        if self.supervised:
            probabilities_unlab = F.softmax(self.module.encoder.encoder_dict['CF'](batch_pred_unlab[pos_graph.nodes()]),dim=-1)
            predictions = probabilities_unlab.argsort(axis=-1)[:,-1]

            fake_nghs = {}
            assigned_molecules = {}
            sampled_reference = []
            prob_dic = {}
            prob = 0
            local_nghs = mfgs[0].srcdata['ngh'][pos_graph.nodes()]
            merged_sum = []
            for l in range(probabilities_unlab.shape[1]):
                lsum = (predictions == l).sum()
                merged_sum.append(lsum)
                merged_genes = bu[predictions == l,:].sum(axis=0)
                averaged_probabilities = F.softmax(probabilities_unlab[predictions == l,:].sum(axis=0),dim=-1)
                assigned_molecules[l] = lsum
                if lsum == 0:
                    merged_genes += 1
                fake_nghs[l] = merged_genes

                ones_tensor = th.ones([int(self.reference.sum(axis=0)[l])])
                s= Multinomial(1,ones_tensor).sample().argsort()[-1] + 35
                sampled_l_ref = Multinomial(int(s), probs= self.reference[:,l]/self.reference[:,l].sum()).sample()
                sampled_reference.append(sampled_l_ref)

                if lsum > 0:
                    merged_genes_ngh = local_nghs[predictions == l,:].sum(axis=0)
                    dist = Multinomial(int(merged_genes.sum()),probs=self.reference[:,l]/self.reference[:,l].sum())
                    pdist = -dist.log_prob(merged_genes)/merged_genes.sum()
                else: 
                    pdist = 0
                prob_dic[l] = pdist
                prob += pdist

            # Introduce reference with sampling
            sampled_reference = th.stack(sampled_reference)
            # Regularize by local nodes
            bone_fight_loss = -F.cosine_similarity(probabilities_unlab @ self.reference.T.to(self.device), local_nghs,dim=1).mean()
            bone_fight_loss = -F.cosine_similarity(probabilities_unlab @ self.reference.T.to(self.device), local_nghs,dim=0).mean()
            # Add Predicted same class nodes together.
            total_counts = th.stack([ms*th.ones(self.reference.shape[0],device=self.device) for ms in merged_sum]) + 1
            probs_per_ct = self.reference/self.reference.sum(axis=0)
            f_probs = []
            for r in range(predictions.shape[0]):
                l,local = predictions[r], local_nghs[r,:]
                lsum = (predictions == l).sum()+local.sum()
                dist = Multinomial(int(lsum), probs=probs_per_ct.T)
                fngh_p= -dist.log_prob(fake_nghs[int(l)] +local)/lsum
                f_probs.append(fngh_p)

            fake_nghs_log_probabilities = th.stack(f_probs)
            fake_nghs_log_probabilities = fake_nghs_log_probabilities.detach()*probabilities_unlab
            prob = fake_nghs_log_probabilities.mean()

            p = local_nghs.sum(axis=1) @ probabilities_unlab
            p = th.log(p/p.sum())
            cells_dist = th.tensor(self.ncells/self.ncells.sum(),dtype=th.float32)
            kl_loss_uniform = self.kl(p,self.p.to(self.device)).sum()*1
            kappa = 2/(1+10**(-1*((1*self.kappa)/200)))-1
            self.kappa += 1
            loss = graph_loss + 1*(kl_loss_uniform+prob+bone_fight_loss)

            for p in prob_dic:
                prob_dic[p]
                self.log(str(p),prob_dic[p],on_step=True)

            self.log('Prob',prob, on_step=True, prog_bar=True,on_epoch=False)
            self.log('KLuniform', kl_loss_uniform, prog_bar=False,on_step=True,on_epoch=False)
            self.log('BFs', bone_fight_loss, prog_bar=True, on_step=True, on_epoch=False)
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('Graph Loss', graph_loss, prog_bar=False, on_step=True, on_epoch=False)

        else:
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

                #if l == 1:
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
        

            self.mean_encoder = nn.Linear(n_hidden, n_hidden)
            self.var_encoder = nn.Linear(n_hidden, n_hidden)
            layers = nn.ModuleList()

            if supervised:
                classifier = Classifier(n_input=n_hidden,
                                        n_labels=n_classes,
                                        softmax=False,
                                        reverse_gradients=False)
            else:
                classifier = None

            if supervised:
                self.norm = F.normalize#DiffGroupNorm(n_hidden,n_classes,classifier) 
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

            self.encoder_dict = nn.ModuleDict({'GS': layers, 
                                                'mean': self.mean_encoder,
                                                'var':self.var_encoder,
                                                'CF':classifier})