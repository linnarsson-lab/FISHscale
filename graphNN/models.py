import torch
from torch_geometric.nn import SAGEConv
import collections
import torch.nn.functional as F
from torch import nn
from typing import Iterable
from torch.distributions import Normal,Poisson
from torch.distributions import Normal, kl_divergence as kl
import pytorch_lightning as pl

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
        num_layers,

        ):

        super().__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            

    def neighborhood_forward(self,x,adjs):
        x = torch.log(x + 1)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.

            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.1, training=self.training)
        return x

    def forward(self,x,pos_x,neg_x,adjs):
        q_m = self.neighborhood_forward(x,adjs)
        q_m_pos = self.neighborhood_forward(pos_x,adjs)
        q_m_neg = self.neighborhood_forward(neg_x,adjs)
    
        pos_loss = F.logsigmoid((q_m * q_m_pos).sum(-1))
        neg_loss = F.logsigmoid(-(q_m * q_m_neg).sum(-1))
        ratio = pos_loss/neg_loss + 1e-8

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()
        
        n_loss = - pos_loss - neg_loss
        return n_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        x,pos,neg,adjs= batch

        loss= self(x,pos,neg,adjs)
        self.log('train_loss', loss)
        return loss
    



