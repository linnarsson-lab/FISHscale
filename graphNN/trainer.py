from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F
from torch_cluster import random_walk
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import negative_sampling,batched_negative_sampling
import numpy as np
from datetime import datetime


class TrainerGNN:

    def __init__(self,
        model,
        graphdata,
        n_epochs=50,
        lr = 0.01,
        mode='autoencoder'):

        """
        Train the SAGE model, either in neighborhood autoencoder, neighborhood embedding or a combination of both.
        Not proven that combination really has an effect.

        Args:
            model (torch.nn.Module): SAGE model
            graphdata (GraphData): Dataset container
            n_epochs (int, optional): number of epochs. Defaults to 50.
            lr (float, optional): learning rate. Defaults to 0.01.
            mode (str, optional): choose from autoencoder, ngh or both. Defaults to 'autoencoder'.
        """        

        self.graphdata = graphdata
        self.train_loader = self.graphdata.train_loader
        self.test_loader = self.graphdata.test_loader
        self.validation_loader = self.graphdata.validation_loader

        self.n_epochs = n_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.x,self.edge_index = self.graphdata.dataset.x.to(self.device), self.graphdata.dataset.edge_index.to(self.device)
        self.mode =mode
        
    def train_step(self):
        self.model.train()

        total_loss = 0
        total_rcl = 0
        total_nl = 0
        for batch_size, n_id, adjs in self.train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            n_id, n_id_pos, n_id_neg = self.sample(n_id,self.train_loader)
            adjs = [adj.to(self.device) for adj in adjs]
            
            #print(ratio*1)
            if self.mode == 'autoencoder':
                rcl = self.model(self.x[n_id], adjs,self.graphdata.local_mean,self.graphdata.local_var)
                loss = rcl #+  n_loss#ratio 

                total_rcl += float(rcl.mean()) * adjs[-1].size[1]
            elif self.mode == 'ngh':
                n_loss,ratio = self.model.neighborhood_loss(self.x[n_id], self.x[n_id_pos],self.x[n_id_neg],adjs)
                loss = n_loss

                total_nl += float(n_loss.mean()) * adjs[-1].size[1]


            elif self.mode == 'both':
                rcl = self.model(self.x[n_id], adjs,self.graphdata.local_mean,self.graphdata.local_var)
                n_loss,ratio = self.model.neighborhood_loss(self.x[n_id], self.x[n_id_pos],self.x[n_id_neg],adjs)
                loss = n_loss + rcl

                total_rcl += float(rcl.mean()) * adjs[-1].size[1]
                total_nl += float(n_loss.mean()) * adjs[-1].size[1]

            #print(ratio)
            loss = loss.mean()
            
            #loss = d_loss#torch.tensor(detach_loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss) * adjs[-1].size[1]
            
            print(total_loss)

        return total_loss /self.graphdata.dataset.num_nodes, total_rcl/self.graphdata.dataset.num_nodes, total_nl/self.graphdata.dataset.num_nodes

    def sample(self, batch,trainer):
        batch = torch.tensor(batch)
        row, col, _ = trainer.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, trainer.adj_t.size(1), (batch.numel(), ),
                                dtype=torch.long)

        return batch,pos_batch,neg_batch


    '''@torch.no_grad()
    def test(self):
        self.model.eval()
        out = self.model.full_forward(self.x, self.edge_index).cpu()

        clf = LogisticRegression()
        clf.fit(out[data.train_mask], data.y[data.train_mask])

        val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
        test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

        return val_acc, test_acc
    '''

    def train(self):
        for epoch in range(1, self.n_epochs+1):
            loss,rcl,nl = self.train_step()
            #val_acc, test_acc = test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, RCL: {rcl:.4f}, Neighborhood Loss: {nl:.4f}')
        self.save_model()

    
    def latent_factor(self):
        self.model.eval()
        embedding = []

        for batch_size, n_id, adjs in self.validation_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            n_id, n_id_pos, n_id_neg = self.sample(n_id,self.validation_loader)
            adjs = [adj.to(self.device) for adj in adjs]
            qn_latent = self.model.encode_neighborhood(self.x[n_id],adjs)
            qz_latent = self.model.encoder_z(qn_latent)
            embedding.append(qz_latent.cpu().detach().numpy())

        X = np.concatenate(embedding)
        return X

    def latent_factor_neighborhood(self):
        self.model.eval()
        embedding = []

        for batch_size, n_id, adjs in self.validation_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            n_id, n_id_pos, n_id_neg = self.sample(n_id,self.validation_loader)
            adjs = [adj.to(self.device) for adj in adjs]
            qz_latent = self.model.encode_neighborhood(self.x[n_id],adjs)
            embedding.append(qz_latent.cpu().detach().numpy())

        X = np.concatenate(embedding)
        return X

    def save_model(self):
        torch.save(self.model.state_dict(), 'SAGEmodel_mode-'+self.mode+datetime.now().strftime("%Y-%m-%d%H-%M-%S"))
