import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, rho_ii=(-5,-3)):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.b_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters(rho_ii)

    def reset_parameters(self, rho_init_range=(-5, -3)):
        nn.init.kaiming_uniform_(self.W_mu, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_mu)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.b_mu, -bound, bound)

        nn.init.uniform_(self.W_rho, *rho_init_range)
        nn.init.uniform_(self.b_rho, *rho_init_range)

    @property
    def W_std(self):
        return torch.logaddexp(torch.zeros(1), self.W_rho)

    @property
    def b_std(self):
        return torch.logaddexp(torch.zeros(1), self.b_rho)

    def forward(self, x):
        W_rand = torch.randn_like(self.W_rho)
        b_rand = torch.randn_like(self.b_rho)

        W = self.W_mu + W_rand * self.W_std
        b = self.b_mu + b_rand * self.b_std

        self.last_W = W
        self.last_b = b

        return F.linear(x, W, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features)


def get_kl_loss(model, num_batches):
    PI = torch.Tensor([0.5])
    SIGMA_1 = torch.exp(torch.Tensor([0.0]))
    SIGMA_2 = torch.exp(torch.Tensor([-6.0]))

    def log_gaussian(x, mu, sigma):
        k = torch.log(2 * np.pi * sigma**2)
        return (-1/2) * (k + (x - mu)**2 / sigma**2)

    def log_q(x, mu, std):
        return torch.sum(log_gaussian(x, mu, std))

    def log_prior(w):
        return torch.sum(torch.logaddexp(
            torch.log(PI) + log_gaussian(w, 0, SIGMA_1),
            torch.log(1 - PI) + log_gaussian(w, 0, SIGMA_2),
        ))

    def layer_log_q(layer):
        W_log_q = log_q(layer.last_W, layer.W_mu, layer.W_std)
        b_log_q = log_q(layer.last_b, layer.b_mu, layer.b_std)
        return W_log_q + b_log_q

    def layer_log_prior(layer):
        W_log_prior = log_prior(layer.last_W)
        b_log_prior = log_prior(layer.last_b)
        return W_log_prior + b_log_prior

    loss = 0.0
    for layer in model:
        if isinstance(layer, BayesLinear):
            loss += layer_log_q(layer) - layer_log_prior(layer)
    return loss / num_batches
