import time

from tqdm import tqdm

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.b_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        MU_INIT_INTERVAL = (-0.2, 0.2)
        RHO_INIT_INTERVAL = (-5, -4)

        nn.init.uniform_(self.W_mu, *MU_INIT_INTERVAL)
        nn.init.uniform_(self.b_mu, *MU_INIT_INTERVAL)
        nn.init.uniform_(self.W_rho, *RHO_INIT_INTERVAL)
        nn.init.uniform_(self.b_rho, *RHO_INIT_INTERVAL)

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


def get_kl_loss(model):
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
            torch.log(PI) * log_gaussian(w, 0, SIGMA_1),
            torch.log(1 - PI) * log_gaussian(w, 0, SIGMA_2),
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
    return loss


def download_mnist():
    import io
    import os
    import requests
    import tarfile

    if os.path.exists('./data/MNIST'):
        return

    print('Downloading MNIST...', end='', flush=True)

    os.makedirs('./data/', exist_ok=True)

    url = 'http://www.di.ens.fr/~lelarge/MNIST.tar.gz'
    mnist_tar_bytes = requests.get(url).content
    mnist_tar_file = io.BytesIO(mnist_tar_bytes)
    tar_obj = tarfile.open(fileobj=mnist_tar_file)
    tar_obj.extractall('./data/')

    print(' done!')

def get_mnist(batch_size=64):
    # Original download source used by datasets.MNIST was
    # not working, so we download dataset from another source
    download_mnist()

    def get_dataloader(train):
        return torch.utils.data.DataLoader(
            datasets.MNIST(
                './data/MNIST',
                train=train,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
            ),
            batch_size=batch_size,
        )

    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    return train_loader, test_loader



def train(
    epochs=30,
    num_samples=5,
    batch_size=128,
    lr=1e-2,
):
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    num_batches = len(train_loader)

    inp_sz = 28**2
    hid_sz = 100
    out_sz = 10
    model = nn.Sequential(
        nn.Flatten(),
        BayesLinear(inp_sz, hid_sz),
        nn.ReLU(),
        BayesLinear(hid_sz, hid_sz),
        nn.ReLU(),
        BayesLinear(hid_sz, out_sz),
    )

    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        num_correct = 0
        for i, (X, y) in enumerate(tqdm(train_loader)):
            loss = 0.0
            for _ in range(num_samples):
                logits = model(X)
                y_pred = torch.argmax(logits, dim=1)

                likelihood_loss = F.cross_entropy(logits, y)
                kl_loss = get_kl_loss(model) / num_batches
                loss += kl_loss + likelihood_loss

                total_loss += loss.item() / num_samples
                num_correct += torch.sum(y_pred == y).item() / num_samples

            loss.backward()
            opt.step()
            opt.zero_grad()

        acc = num_correct / len(train_loader.dataset)

        print('----------')
        print(f'epoch: {epoch}')
        print(f'loss: {total_loss}')
        print(f'acc : {acc}')

if __name__ == '__main__':
    train()
