import torch
from torch_geometric.nn import SAGEConv
import collections
import torch.nn.functional as F
from torch import nn
from typing import Iterable
from torch.distributions import Normal,Poisson
from torch.distributions import Normal, kl_divergence as kl

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()


        self.encoder_library = Encoder(in_channels,1)
        self.encoder_z = Encoder(hidden_channels,hidden_channels)

        self.decoder = Decoder(hidden_channels,in_channels)

        self.mean_encoder = nn.Linear(hidden_channels, hidden_channels)
        self.var_encoder = nn.Linear(hidden_channels, hidden_channels)

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

        
    def forward(self, x, adjs,local_l_mean,local_l_var):
        target_nodes = adjs[-1].size[1]
        local_l_mean = torch.reshape(torch.ones([target_nodes],dtype=torch.float32)*local_l_mean,[target_nodes,1]).cpu()
        local_l_var = torch.reshape(torch.ones([target_nodes],dtype=torch.float32)*local_l_var,[target_nodes,1]).cpu()
        ql_m, ql_v,library = self.encoder_library(torch.log(x[:target_nodes]+1))
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)),Normal(local_l_mean, torch.sqrt(local_l_var)),).sum(dim=1)
        
        x2 = self.encode_neighborhood(x,adjs)
        qz_m,qz_v,z_latent = self.encoder_z(x2)
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)

        px_scale,px_rate,px_r = self.decoder(z_latent, library)
        rcl = -self.log_poisson(x[:target_nodes],px_rate).sum(dim=-1)

        return rcl+kl_divergence_z+kl_divergence_l

    def encode_neighborhood(self,x,adjs):
        x = torch.log(x + 1)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.

            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.1, training=self.training)
        return x

    def neighborhood_loss(self,x,pos_x,neg_x,adjs):
        q_m = self.encode_neighborhood(x,adjs)
        q_m_pos = self.encode_neighborhood(pos_x,adjs)
        q_m_neg = self.encode_neighborhood(neg_x,adjs)
    
        pos_loss = F.logsigmoid((q_m * q_m_pos).sum(-1))
        neg_loss = F.logsigmoid(-(q_m * q_m_neg).sum(-1))
        ratio = pos_loss/neg_loss + 1e-8

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()
        
        n_loss = - pos_loss - neg_loss
        return n_loss,ratio

    def log_nb_positive(self, x, mu, theta, eps=1e-8):
        log_theta_mu_eps = torch.log(theta + mu + eps)
        negative_likelihood = (theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )

        return negative_likelihood

    def log_poisson(self, x, mu, eps=1e-8):
        negative_likelihood = Poisson(mu).log_prob(x)

        return negative_likelihood
    

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.1, training=self.training)

        q_m = self.mean_encoder(x)
        q_v = torch.exp(self.var_encoder(x)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m


class FCLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 24,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_relu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out, bias=bias),
                            # Below, 0.01 and 0.001 are the default values for `momentum` and `eps` from
                            # the tensorflow implementation of batch norm; we're using those settings
                            # here too so that the results match our old tensorflow code. The default
                            # setting from pytorch would probably be fine too but we haven't tested that.
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.ReLU() if use_relu else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :param instance_id: Use a specific conditional instance normalization (batchnorm)
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(
            cat_list
        ), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (
                n_cat and cat is None
            ), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 24,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )
        self.px_r_decoder = nn.Linear(n_hidden, n_output)


    def forward(self, z,library):
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_rate = px_scale*torch.exp(library)
        px_r = torch.exp(self.px_r_decoder(px))
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        # torch.clamp( , max=12)

        return px_scale,px_rate,px_r

class Encoder(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor):

        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent