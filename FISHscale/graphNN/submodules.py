import torch as th
from torch import nn
from torch import Tensor
from torch.nn import Linear, BatchNorm1d
from torch.autograd import Function

class Classifier(nn.Module):
    """Basic fully-connected NN classifier
    """

    def __init__(
        self,
        n_input,
        n_hidden=48,
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
        layers = [
            nn.Linear(n_input, n_labels),]

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

class PairNorm(th.nn.Module):
    r"""Applies pair normalization over node features as described in the
    `"PairNorm: Tackling Oversmoothing in GNNs"
    <https://arxiv.org/abs/1909.12223>`_ paper

    .. math::
        \mathbf{x}_i^c &= \mathbf{x}_i - \frac{1}{n}
        \sum_{i=1}^n \mathbf{x}_i \\

        \mathbf{x}_i^{\prime} &= s \cdot
        \frac{\mathbf{x}_i^c}{\sqrt{\frac{1}{n} \sum_{i=1}^n
        {\| \mathbf{x}_i^c \|}^2_2}}

    Args:
        scale (float, optional): Scaling factor :math:`s` of normalization.
            (default, :obj:`1.`)
        scale_individually (bool, optional): If set to :obj:`True`, will
            compute the scaling step as :math:`\mathbf{x}^{\prime}_i = s \cdot
            \frac{\mathbf{x}_i^c}{{\| \mathbf{x}_i^c \|}_2}`.
            (default: :obj:`False`)
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, scale: float = 1., scale_individually: bool = False,
                 eps: float = 1e-5):
        super(PairNorm, self).__init__()
        from torch_scatter import scatter
        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps

    def forward(self, x, batch=None):
        scale = self.scale

        if batch is None:
            x = x - x.mean(dim=0, keepdim=True)

            if not self.scale_individually:
                return scale * x / (self.eps + x.pow(2).sum(-1).mean()).sqrt()
            else:
                return scale * x / (self.eps + x.norm(2, -1, keepdim=True))

        else:
            mean = scatter(x, batch, dim=0, reduce='mean')
            x = x - mean.index_select(0, batch)

            if not self.scale_individually:
                return scale * x / th.sqrt(self.eps + scatter(
                    x.pow(2).sum(-1, keepdim=True), batch, dim=0,
                    reduce='mean').index_select(0, batch))
            else:
                return scale * x / (self.eps + x.norm(2, -1, keepdim=True))


    def __repr__(self):
        return f'{self.__class__.__name__}()'

class DiffGroupNorm(th.nn.Module):
    r"""The differentiable group normalization layer from the `"Towards Deeper
    Graph Neural Networks with Differentiable Group Normalization"
    <https://arxiv.org/abs/2006.06972>`_ paper, which normalizes node features
    group-wise via a learnable soft cluster assignment

    .. math::

        \mathbf{S} = \text{softmax} (\mathbf{X} \mathbf{W})

    where :math:`\mathbf{W} \in \mathbb{R}^{F \times G}` denotes a trainable
    weight matrix mapping each node into one of :math:`G` clusters.
    Normalization is then performed group-wise via:

    .. math::

        \mathbf{X}^{\prime} = \mathbf{X} + \lambda \sum_{i = 1}^G
        \text{BatchNorm}(\mathbf{S}[:, i] \odot \mathbf{X})

    Args:
        in_channels (int): Size of each input sample :math:`F`.
        groups (int): The number of groups :math:`G`.
        lamda (float, optional): The balancing factor :math:`\lambda` between
            input embeddings and normalized embeddings. (default: :obj:`0.01`)
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
    """
    def __init__(self, in_channels, groups, function=None,lamda=0.00001, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(DiffGroupNorm, self).__init__()

        self.in_channels = in_channels
        self.groups = groups
        self.lamda = lamda
        if type(function) == type(None):
            self.lin = Linear(in_channels, groups, bias=False)
        else:
            self.lin = function
        self.norm = BatchNorm1d(groups * in_channels, eps, momentum, affine,
                                track_running_stats)

        self.reset_parameters()

    def reset_parameters(self):
        #self.lin.reset_parameters()
        self.norm.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        F, G = self.in_channels, self.groups

        s = self.lin(x).softmax(dim=-1)  # [N, G]
        out = s.unsqueeze(-1) * x.unsqueeze(-2)  # [N, G, F]
        out = self.norm(out.view(-1, G * F)).view(-1, G, F).sum(-2)  # [N, F]

        return x + self.lamda * out


    @staticmethod
    def group_distance_ratio(x: Tensor, y: Tensor, eps: float = 1e-5) -> float:
        r"""Measures the ratio of inter-group distance over intra-group
        distance

        .. math::
            R_{\text{Group}} = \frac{\frac{1}{(C-1)^2} \sum_{i!=j}
            \frac{1}{|\mathbf{X}_i||\mathbf{X}_j|} \sum_{\mathbf{x}_{iv}
            \in \mathbf{X}_i } \sum_{\mathbf{x}_{jv^{\prime}} \in \mathbf{X}_j}
            {\| \mathbf{x}_{iv} - \mathbf{x}_{jv^{\prime}} \|}_2 }{
            \frac{1}{C} \sum_{i} \frac{1}{{|\mathbf{X}_i|}^2}
            \sum_{\mathbf{x}_{iv}, \mathbf{x}_{iv^{\prime}} \in \mathbf{X}_i }
            {\| \mathbf{x}_{iv} - \mathbf{x}_{iv^{\prime}} \|}_2 }

        where :math:`\mathbf{X}_i` denotes the set of all nodes that belong to
        class :math:`i`, and :math:`C` denotes the total number of classes in
        :obj:`y`.
        """
        num_classes = int(y.max()) + 1

        numerator = 0.
        for i in range(num_classes):
            mask = y == i
            dist = th.cdist(x[mask].unsqueeze(0), x[~mask].unsqueeze(0))
            numerator += (1 / dist.numel()) * float(dist.sum())
        numerator *= 1 / (num_classes - 1)**2

        denominator = 0.
        for i in range(num_classes):
            mask = y == i
            dist = th.cdist(x[mask].unsqueeze(0), x[mask].unsqueeze(0))
            denominator += (1 / dist.numel()) * float(dist.sum())
        denominator *= 1 / num_classes

        return numerator / (denominator + eps)


    def __repr__(self):
        return '{}({}, groups={})'.format(self.__class__.__name__,
                                          self.in_channels, self.groups)