import torch
from torch_geometric.nn import SAGEConv, GATConv
import collections
import torch.nn.functional as F
from torch import nn
from typing import Iterable
from torch.distributions import Normal,Poisson
from torch.distributions import Normal, kl_divergence as kl
import pytorch_lightning as pl
import torchmetrics
import math

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

class SAGE(pl.LightningModule):
    """
    GraphSAGE based model in combination with scvi variational autoencoder.

    SAGE will learn to encode neighbors to allow either the reconstruction of the original nodes data helped by neighbor data or 
    to generate similar embedding for closeby nodes (i.e. regionalization).

 
    """     

    def __init__(self, 
        in_channels :int, 
        hidden_channels:int,
        num_layers:int=2,
        normalize:bool=True,
        apply_normal_latent:bool=False,
        supervised:bool=False,
        output_channels:int=448,
        loss_fn:str='sigmoid',
        max_lambd:int=100000,

        ):
        """
        __init__ [summary]

        [extended_summary]

        Args:
            in_channels (int): [description]
            hidden_channels (int): [description]
            num_layers (int, optional): [description]. Defaults to 2.
            normalize (bool, optional): [description]. Defaults to True.
            apply_normal_latent (bool, optional): [description]. Defaults to False.
            supervised (bool, optional): [description]. Defaults to False.
            output_channels (int, optional): [description]. Defaults to 448.
        """        

        super().__init__()
        self.save_hyperparameters()

        self.num_layers = num_layers
        self.normalize = normalize
        self.convs = torch.nn.ModuleList()
        self.apply_normal_latent = apply_normal_latent
        self.supervised= supervised
        self.loss_fn = loss_fn
        self.progress = 0
        self.max_lambd = max_lambd
        self.automatic_optimization = False

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            # L2 regularization
            if i == num_layers-1:
                self.convs.append(SAGEConv(in_channels, hidden_channels,normalize=False))
                #self.convs.append(GATConv(in_channels, hidden_channels, heads=8, dropout=0.1,concat=False))
            else:
                self.convs.append(SAGEConv(in_channels, hidden_channels,normalize=self.normalize))
                #self.convs.append(GATConv(in_channels, hidden_channels, heads=8, dropout=0.1,concat=False))

        '''        
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))'''

        if self.apply_normal_latent:
            self.mean_encoder = nn.Linear(hidden_channels, hidden_channels)
            self.var_encoder = nn.Linear(hidden_channels, hidden_channels)

        if self.supervised:
            self.classifier = Classifier(n_input=hidden_channels,n_labels=output_channels,softmax=False)
            self.train_acc = torchmetrics.Accuracy()
                
    def neighborhood_forward(self,x,adjs):
        x = torch.log(x + 1)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.

            x = self.convs[i]((x, x_target), edge_index)
            #x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                #x = self.bns[i](x)
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)

        if self.apply_normal_latent:
            q_m = self.mean_encoder(x)
            q_v = torch.exp(self.var_encoder(x)) + 1e-4
            x = reparameterize_gaussian(q_m, q_v)
        else:
            q_m = 0
            q_v = 0

        return x, q_m, q_v

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def forward(self,x,adjs,classes=None):
        # Embedding sampled nodes
        z, qm, qv = self.neighborhood_forward(x, adjs)
        z, z_pos, z_neg = z.split(z.size(0) // 3, dim=0)
        if type(qm) != int:
            qm, qm_pos, qm_neg = qm.split(qm.size(0) // 3, dim=0)
            qv, qv_pos, qv_neg = qv.split(qv.size(0) // 3, dim=0)
        # Embedding for neighbor nodes of sample nodes

        if self.loss_fn == 'sigmoid':
            pos_loss = -F.logsigmoid((z * z_pos).sum(-1))
            neg_loss = -F.logsigmoid(-(z * z_neg).sum(-1))
        elif self.loss_fn == 'cosine':
            pos_loss = -torch.cosine_similarity(z,z_pos)
            neg_loss = torch.cosine_similarity(z,z_neg)


        lambd = 2 / (1 + math.exp(-10 * self.progress/self.max_lambd)) - 1
        self.progress += 1
        pos_loss = pos_loss.mean() #* lambd
        neg_loss = neg_loss.mean() #* 100
        n_loss = pos_loss + neg_loss

        # KL Divergence
        if self.apply_normal_latent:
            mean = torch.zeros_like(qm)
            scale = torch.ones_like(qv)
            kl_divergence_z = kl(Normal(qm, torch.sqrt(qv)), Normal(mean, scale)).sum(dim=1)
            n_loss = n_loss + kl_divergence_z.mean()
        
        # Add loss if trying to reconstruct cell types
        if type(classes) != type(None):
            prediction = self.classifier(z)
            cce = torch.nn.CrossEntropyLoss()
            classifier_loss = cce(prediction,classes)
            #self.train_acc(y_hat.softmax(dim=-1), y)
            n_loss += classifier_loss #* 10
            #print(supervised_loss)
            self.log('Classifier Loss',classifier_loss)
            #self.train_acc(prediction.softmax(dim=-1),F.one_hot(classes,num_classes=prediction.shape[1]))
            self.train_acc(prediction.argsort(axis=-1)[:,-1],classes)
            self.log('train_acc', self.train_acc, prog_bar=True, on_step=True)
        else:
            n_loss = n_loss #* 10
            
        return n_loss, pos_loss, neg_loss

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.01)#,weight_decay=5e-4)

        pos_opt = torch.optim.Adam(self.parameters(), lr=0.001)
        neg_opt = torch.optim.Adam(self.parameters(), lr=0.001)
        return pos_opt, neg_opt


    def training_step(self, batch, batch_idx,optimizer_idx):
        x,adjs,c = batch['unlabelled']
        loss,pos_loss,neg_loss = self(x, adjs, c)
        p_opt,n_opt = self.optimizers()
        p_opt.zero_grad()
        self.manual_backward(pos_loss)
        p_opt.step()

        loss,pos_loss,neg_loss = self(x, adjs, c)
        n_opt.zero_grad()
        self.manual_backward(neg_loss)
        n_opt.step()

        if 'labelled' in batch:
            x, adjs, c = batch['labelled']
            loss_labelled, _, _ = self(x, adjs, c)
            self.log('labelled_loss',loss_labelled)
            loss += loss_labelled

        self.log('Positive Loss', pos_loss,on_step=True, on_epoch=True,prog_bar=True)
        self.log('Negative Loss', neg_loss,on_step=True, on_epoch=True,prog_bar=True)
        self.log('train_loss', loss,on_step=True, on_epoch=True,prog_bar=True)
        
        #return loss

    def validation_step(self, batch, batch_idx):
        x, adjs, c = batch
        loss = self(x, adjs, c)
        self.log('val_loss', loss)
        return loss
    
# Decoder
class DecoderSCVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 48,
        use_batch_norm: bool=True,
        use_relu:bool=True,
        dropout_rate: float=0.1,
        bias: bool=True,
        softmax:bool = True,
    ):
        super().__init__()

        self.px_decoder = nn.Sequential(
                            nn.Linear(n_input , n_hidden, bias=bias),
                            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            nn.ReLU() if use_relu else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None)

        if softmax:
            self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output))
        else:
            self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output),
                nn.Softmax(dim=-1))

    def forward(
        self, z: torch.Tensor
    ):
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)
        px= self.px_scale_decoder(px)
        return px

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
                            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            nn.ReLU() if use_relu else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None),
            nn.Linear(n_hidden, n_labels),]

        if softmax:
            layers.append(nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(*layers,nn.ReLU())

    def forward(self, x):
        return F.log_softmax(self.classifier(x),dim=-1)
