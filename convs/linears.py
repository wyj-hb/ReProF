'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from timm.layers.weight_init import trunc_normal_
class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


class CosineLinearForDS(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinearForDS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.use_RP=False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn=input
                #inn=torch.bmm(input[:,0:100].unsqueeze(-1), input[:,0:100].unsqueeze(-2)).flatten(start_dim=1) #interaction terms instead of RP
            out = F.linear(inn,self.weight)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.use_RP=False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn=input

            out = F.linear(inn,self.weight)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': reduce_proxies(out1['logits'], self.nb_proxy),
            'new_scores': reduce_proxies(out2['logits'], self.nb_proxy),
            'logits': out
        }


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


from collections import defaultdict

class KLDA:
    def __init__(self, num_classes, d, D, sigma, seed, device):
        """
        Args:
            num_classes (int): Total number of classes.
            d (int): Input feature dimension.
            D (int): RFF dimension.
            sigma (float): Bandwidth parameter for RBF kernel.
            seed (int): Seed for RFF initialization to ensure different transformations.
            device (torch.device): Device to run the models on.
        """
        self.num_classes = num_classes
        self.d = d
        self.D = D
        self.sigma = sigma
        self.seed = seed
        self.device = device
        torch.manual_seed(self.seed)
        self.omega = torch.normal(
            0, 
            torch.sqrt(torch.tensor(2 * self.sigma, dtype=torch.float32)),
            (self.d, self.D)
        ).to(self.device)
        self.b = (torch.rand(self.D) * 2 * torch.pi).to(self.device)

        self.class_means = defaultdict(lambda: torch.zeros(self.D, device=self.device))
        self.class_counts = defaultdict(int)

        self.sigma = torch.zeros((self.D, self.D), device=self.device)
        self.sigma_inv = None
        self.class_mean_matrix = None

    def _compute_rff(self, X):
        """
        Computes the Random Fourier Features for input data X.
        Args:
            X (torch.Tensor): Input feature tensor of shape (n_samples, d).
        Returns:
            torch.Tensor: Transformed feature tensor of shape (n_samples, D).
        """
        scaling_factor = torch.sqrt(torch.tensor(2.0 / self.D, dtype=torch.float32, device=self.device))
        return scaling_factor * torch.cos(X @ self.omega + self.b)

    def batch_update(self, X, y):
        """
        Updates the model with a batch of data for a specific class.
        Args:
            X (torch.Tensor): Feature tensor of shape (n_samples, d).
            y (int): Class label.
        """
        X = X.to(self.device)
        n = X.size(0)
        phi_X = self._compute_rff(X)  # Shape: (n, D)
        phi_X_mean = torch.mean(phi_X, dim=0)

        # Update class mean
        previous_count = self.class_counts[y]
        self.class_counts[y] += n
        self.class_means[y] = (self.class_means[y] * previous_count + phi_X_mean * n) / self.class_counts[y]

        # Update covariance matrix sigma
        centered_phi_X = phi_X - self.class_means[y]
        self.sigma += centered_phi_X.t() @ centered_phi_X

    def fit(self):
        """
        Finalizes the model by computing the inverse of the covariance matrix
        and stacking class means into a matrix for efficient distance computation.
        """
        self.sigma_inv = torch.pinverse(self.sigma)
        self.num_classes = len(self.class_means)
        self.class_mean_matrix = torch.stack([self.class_means[i] for i in range(self.num_classes)]).to(self.device)

    def get_logits(self, x):
        """
        Computes the logits for all the classes.
        Args:
            x (torch.Tensor): Feature tensor of shape (d,).
        Returns:
            torch.Tensor: Logits for each class of shape (num_classes,).
        """
        x = x.to(self.device)
        phi_x = self._compute_rff(x)  # Shape: (1, D)
        diff = self.class_mean_matrix.unsqueeze(0) - phi_x.unsqueeze(1)  # Shape: (batch_size, num_classes, D)
        logits = -torch.sum((diff @ self.sigma_inv) * diff, dim=2)  # Shape: (batch_size, num_classes)     # Shape: (num_classes, D)
        return logits

class KLDA_E:
    def __init__(self, num_classes, d, D, sigma, num_ensembles, seed, device=None):
        """
        Args:
            num_classes (int): Total number of classes.
            d (int): Input feature dimension.
            D (int): RFF dimension.
            sigma (float): Bandwidth parameter for RBF kernel.
            num_ensembles (int): Number of models in the ensemble.
            seed (int): Base seed for reproducibility.
            device (torch.device, optional): Device to run the models on. Defaults to CUDA if available.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_ensembles = num_ensembles
        self.device = device
        self.models = [
            KLDA(num_classes, d, D, sigma, seed=seed+i, device=self.device) for i in range(self.num_ensembles)
        ]

    def batch_update(self, X, y):
        for model in self.models:
            model.batch_update(X, y)

    def fit(self):
        for model in self.models:
            model.fit()

    def predict(self, x):
        total_probabilities = torch.zeros(x.shape[0],self.models[0].num_classes, device=self.device)
        for model in self.models:
            logits = model.get_logits(x)
            probs = torch.softmax(logits, dim=1)
            total_probabilities += probs
        res = torch.sort(total_probabilities,dim = 1,descending=True).indices

        return res
class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=True))
        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02) 

    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))
            
        _fc = nn.Linear(self.embed_dim, nb_classes, bias=False)
        trunc_normal_(_fc.weight, std=.02)
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad=False

        self.heads.append(new_head)

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(1*(F.linear(F.normalize(fc_inp, p=2, dim=1),F.normalize(self.heads[ti][0].weight, p=2, dim=1))))
        out = {'logits': torch.cat(out, dim=1)}
        return out
# * conv_layer
def conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            nn.Tanh()
        )
    else:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            # nn.Tanh()
        )
    return res

class AC_Linear(nn.Module):
    def __init__(self, in_features, Hidden, out_dim):
        super(AC_Linear, self).__init__()
        self.in_features = in_features
        self.hidden = Hidden
        self.out_features = out_dim
        bias_fe = False

        self.fc = nn.Sequential(nn.Linear(self.in_features, self.hidden, bias=bias_fe),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden, self.out_features, bias=False))


    def forward(self, input):
        hidden_feature = self.fc[:2](input)
        out = self.fc(input)

        return {'hidden_feature': hidden_feature, 'logits': out}

class CosineLinear2(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'train_logits': out}