import torch as th
from torch import nn
from torch import Tensor
from torch.nn import Linear, BatchNorm1d
from torch.autograd import Function
from torch.distributions import Distribution, Gamma, Poisson, constraints

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
    def __init__(self, in_channels, groups, function=None,lamda=1e-6, eps=1e-5, momentum=0.1,
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


def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion
    Parameters
    ----------
    mu :
        mean of the NB distribution.
    theta :
        inverse overdispersion.
    eps :
        constant used for numerical log stability. (Default value = 1e-6)
    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    assert (mu is None) == (
        theta is None
    ), "If using the mu/theta NB parameterization, both parameters must be specified"
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


def _convert_counts_logits_to_mean_disp(total_count, logits):
    """NB parameterizations conversion
    Parameters
    ----------
    total_count :
        Number of failures until the experiment is stopped.
    logits :
        success logits.
    Returns
    -------
    type
        the mean and inverse overdispersion of the NB distribution.
    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta


class NegativeBinomial(Distribution):
    r"""Negative Binomial(NB) distribution using two parameterizations:
    - (`total_count`, `probs`) where `total_count` is the number of failures
        until the experiment is stopped
        and `probs` the success probability.
    - The (`mu`, `theta`) parameterization is the one used by scVI. These parameters respectively
    control the mean and overdispersion of the distribution.
    `_convert_mean_disp_to_counts_logits` and `_convert_counts_logits_to_mean_disp` provide ways to convert
    one parameterization to another.
    Parameters
    ----------
    Returns
    -------
    """
    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: th.Tensor = None,
        probs: th.Tensor = None,
        logits: th.Tensor = None,
        mu: th.Tensor = None,
        theta: th.Tensor = None,
        validate_args=True,
    ):
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation for more information."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            mu, theta = _convert_counts_logits_to_mean_disp(total_count, logits)
        else:
            mu, theta = broadcast_all(mu, theta)
        self.mu = mu
        self.theta = theta
        super().__init__(validate_args=validate_args)

    def sample(self, sample_shape=th.Size()):
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = th.clamp(p_means, max=1e8)
        counts = Poisson(
            l_train
        ).sample()  # Shape : (n_samples, n_cells_batch, n_genes)
        return counts

    def log_prob(self, value):
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                )
        return log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)

    def _gamma(self):
        concentration = self.theta
        rate = self.theta / self.mu
        # Important remark: Gamma is parametrized by the rate = 1/scale!
        gamma_d = Gamma(concentration=concentration, rate=rate)
        return gamma_d
