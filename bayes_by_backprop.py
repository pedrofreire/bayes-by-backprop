import time

from tqdm import tqdm

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = 'cpu'

class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features,rho_ii=(-5,-3)):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.b_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters(rho_ii)

    def reset_parameters(self,rho_ii=(-5, -3)):
        nn.init.kaiming_uniform_(self.W_mu, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_mu)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.b_mu, -bound, bound)

        nn.init.uniform_(self.W_mu, -0.2, 0.2)
        nn.init.uniform_(self.b_mu, -0.2, 0.2)
        nn.init.uniform_(self.W_rho, *rho_ii)
        nn.init.uniform_(self.b_rho, *rho_ii)

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
    epochs=300,
    num_samples=5,
    batch_size=128,
    lr=1e-3,
    hidden_size=100,
    kl_const=1e-3,
):
    train_loader, test_loader = get_mnist(batch_size=batch_size)

    input_size = 28**2
    output_size = 10
    model = nn.Sequential(
        nn.Flatten(),
        BayesLinear(input_size, hidden_size),
        nn.ReLU(),
        BayesLinear(hidden_size, hidden_size),
        nn.ReLU(),
        BayesLinear(hidden_size, output_size),
        nn.LogSoftmax(dim=-1),
    )

    opt = optim.Adam(model.parameters(), lr=lr)

    def run_epoch(dataloader, train, epoch):
        num_batches = len(dataloader)

        total_loss = 0.0
        total_lh_loss = 0.0
        total_kl_loss = 0.0
        num_correct = 0
        for i, (X, y) in enumerate(tqdm(dataloader, position=0, leave=True)):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            cur_batch_size = y.shape[0]
            log_probs = torch.zeros(num_samples, cur_batch_size, output_size)
            kl_losses = torch.zeros(num_samples)
            for i in range(num_samples):
                log_probs[i] = model(X)
                kl_losses[i] = get_kl_loss(model, num_batches)

            y_pred = torch.argmax(log_probs.mean(dim=0), dim=1)
            likelihood_loss = F.cross_entropy(log_probs.mean(dim=0), y)
            kl_loss = kl_losses.mean()
            loss = kl_const * kl_loss + likelihood_loss

            if train:
                loss.backward()
                opt.step()
                opt.zero_grad()

            total_loss += loss.item()
            total_lh_loss += likelihood_loss.item()
            total_kl_loss += kl_loss.item()
            num_correct += torch.sum(y_pred == y).item()

        acc = num_correct / len(dataloader.dataset)

        if train:
            print('---train---')
        else:
            print('---test----')
        print(f'epoch: {epoch}')
        print(f'loss: {total_loss}')
        print(f'lh_loss: {total_lh_loss}')
        print(f'kl_loss: {total_kl_loss}')
        print(f'acc : {acc}')


    for epoch in range(epochs):
        run_epoch(train_loader, train=True, epoch=epoch)
        if epoch % 3 == 0:
            run_epoch(test_loader, train=False, epoch=epoch)

#train()