import matplotlib.pyplot as plt
import numpy as np


import torch
from torch import nn, optim
from torch.nn import functional as F

from bayes_by_backprop import BayesLinear, get_kl_loss

class Reg_DS:
    def __init__(self,N=1000,sig=0.02,norm=True):
        t=torch.linspace(0,0.5,N)
        eps=torch.randn(N)*sig
        y=t+0.3*(torch.sin(2*np.pi*(t+eps))+torch.sin(4*np.pi*(t+eps)))+eps
        self.norm=norm
        if norm:
            self.tmean,self.ymean=t.mean(),y.mean()
            self.tstd,self.ystd=t.std(),y.std()

            t=(t-t.mean())/t.std()
        self.t=t
        self.y=y


    def __len__(self):
        return len(self.y)
    def __getitem__(self,i):
        #if self.norm:
        #    return self.t[i]*self.tstd+self.tmean,self.y[i]*self.ystd+self.ymean
        return self.t[i],self.y[i]
    
    
def train(
    epochs=30,
    num_samples=5,
    batch_size=64,
    lr=1e-3,bayesian=True,BETA=1e-1,
    hid_sz=100,num_layers=3,log=[]
):
    train_loader=torch.utils.data.DataLoader(Reg_DS(norm=True),batch_size,shuffle=True)
    num_batches=len(train_loader)
    inp_sz = 1
    out_sz = 1
    linear = BayesLinear if bayesian else nn.Linear
    model = nn.Sequential(*([
        linear(inp_sz, hid_sz),
        nn.ReLU()]+
        (num_layers-2)*[linear(hid_sz, hid_sz),
        nn.ReLU()]+
        [linear(hid_sz, out_sz)])
    )



    
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        num_correct = 0
        for i, (X, y) in enumerate(train_loader):
            kl_losses=torch.zeros(num_samples)
            reg_losses=torch.zeros(num_samples)
            for ns in range(num_samples):
                reg_loss=((model(X[:,None]).reshape(-1)-y)**2).mean()
                kl_loss = get_kl_loss(model, num_batches)
                loss = BETA * kl_loss + reg_loss
                loss.backward()
                log.append((reg_loss.item(),kl_loss.item(),loss.item()))
                total_loss += loss.item() /num_samples
                opt.step()
                opt.zero_grad()
        if epoch:
            #print('----------')
            print(f'epoch: {epoch}')
            print(f'loss: {total_loss}')
    return model,log

                

if __name__=='__main__':
    log=[]
    model,log=train(100,lr=1e-2,bayesian=True,num_samples=5,BETA=1e-5,log=log,hid_sz=100)
    N=1000
    NS=100
    samples=torch.zeros(NS,N)
    ds=Reg_DS(norm=True)
    t=torch.linspace(-0.2,0.7,N)

    ymean,ystd=ds.y.mean(),ds.y.std()
    for i in range(NS):
        samples[i]=model(((t-ds.tmean)/ds.tstd)
                          [:,None]).reshape(-1)
    #samples=samples*ystd+ymean
    sort=torch.sort(samples,axis=0)[0].cpu().detach()
    mu,std=samples.mean(axis=0).cpu().detach(),    samples.std(axis=0).cpu().detach()
    ax=plt.subplot()
    ax.plot(t.cpu(),mu,c='b')
    plt.scatter((ds.t*ds.tstd+ds.tmean).cpu().detach(),ds.y.cpu(),marker='x')

    ax.fill_between(t.cpu(),sort[25],sort[75],color='g',alpha=0.2)
    ax.fill_between(t.cpu(),sort[36],sort[62],color='y',alpha=0.4)

    plt.show()
