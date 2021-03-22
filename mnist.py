from collections import defaultdict

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms


from bayes_by_backprop import BayesLinear, get_kl_loss

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
    hidden_size=400,
    kl_const=1e-6,
    verbose=True,
):
    train_log = defaultdict(list)
    def log_vars(lcls):
        def printv(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        train = lcls['train']
        epoch = lcls['epoch']

        prefix = 'train' if train else 'test'

        printv(f'epoch : {epoch}')
        for varname in (
            'total_loss',
            'total_lh_loss',
            'total_kl_loss',
            'acc',
        ):
            label = f'{prefix}/{varname}'
            value = lcls[varname]
            train_log[label].append((epoch, value))
            printv(f'{label} : {value:.4f}')


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

    def run_epoch(train, epoch):
        dataloader = train_loader if train else test_loader
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

        log_vars(locals())

    for epoch in range(epochs):
        run_epoch(train=True, epoch=epoch)
        if epoch % 3 == 0:
            run_epoch(train=False, epoch=epoch)

    return train_log

def save_results(train_log):
    import pickle
    with open('results.pkl', 'wb') as f:
        pickle.dump(train_log, f)

def plot_results():
    import matplotlib.pyplot as plt
    import pickle

    with open('results.pkl', 'rb') as f:
        train_log = pickle.load(f)

    test_acc_points = train_log['test/acc']
    test_epochs, test_accs = np.array(test_acc_points).T
    test_errs = 1 - test_accs

    plt.plot(test_epochs, test_errs, label='test error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        DEVICE = 'cpu'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--no_train', dest='train', action='store_false')
    parser.add_argument('--plot', default=False, action='store_true')

    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-s', '--num_samples', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-hs', '--hidden_size', type=int, default=100)
    parser.add_argument('-kl', '--kl_const', type=float, default=1e-3)

    args = parser.parse_args()

    if args.train:
        train_log = train(
            epochs=args.epochs,
            num_samples=args.num_samples,
            lr=args.learning_rate,
            hidden_size=args.hidden_size,
            kl_const=args.kl_const,
        )
        save_results(train_log)
    if args.plot:
        plot_results()
