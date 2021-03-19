from random import choice
from bayes_by_backprop import BayesLinear
import pandas as pd   
import numpy as np

BayesLinear=bayes_by_backprop.BayesLinear



class Mushroom_Bandit:
    def __init__(self):
        df=pd.read_csv('data/agaricus-lepiota.data',header=None)
        y=df[0].values
        y=np.array([1 if t=='p' else 0 for t in y])
        x=pd.get_dummies(df.iloc[:,1:]).values
        self.x=torch.tensor(x)
        self.y=torch.tensor(y)
        self.cum_reg=[0]
        self.plays=[]
    
    def get_ctx(self):
        #self.i=torch.randint(0,len(self.x),(1))
        self.i=np.random.randint(0,len(self.x))
        return self.x[self.i]
    
    def get_reward(self,a):
        y=self.y[self.i]
        mer=0 if y==1 else 5
        
        self.plays.append((1-a,y))
        if a==0: ##don't eat
            self.cum_reg.append(mer+self.cum_reg[-1])
            return 0
        
        if y==0: ##eat and not poisonous
            self.cum_reg.append(self.cum_reg[-1])
            return 5
        
        ##eat and poisonous
        self.cum_reg.append(self.cum_reg[-1]+15)
        
        p=torch.rand(1)
        if p>=0.5:
            return 5
        
        return -35
    


    
  
class Agent:
    def __init__(self,bandit,num_batches):
        self.buffer=[]
        self.bandit=bandit
        self.BUFFER_SIZE=64*num_batches
    
    
    def update():
        pass
        
    
    def get_action(self,ctx):
        pass
    
    def step(self):
        ctx=self.bandit.get_ctx()
        if len(self.buffer)<self.BUFFER_SIZE:
            action=choice([0,1])
        else:
            action=self.get_action(ctx.to(dtype=torch.float))
        
        rwd=self.bandit.get_reward(action)
        
        if len(self.buffer)>=self.BUFFER_SIZE:
            self.buffer.pop(0)
        
        self.buffer.append((torch.tensor(ctx,dtype=torch.float),
                            torch.tensor(action,dtype=torch.long),
                            torch.tensor(rwd,dtype=torch.float)))
        
        if len(self.buffer)>=self.BUFFER_SIZE:
            #idxs=torch.randperm(len(self.buffer)).cpu().numpy()[:64]
            #actions=torch.tensor([self.buffer[i][1] for i in idxs],dtype=torch.long)
            #ctxs=torch.stack([self.buffer[i][0] for i in idxs]).to(dtype=torch.float)
            #rwds=torch.tensor([self.buffer[i][2] for i in idxs],dtype=torch.float)
            
            self.update()

class Eps_Greedy(Agent):
    def __init__(self,bandit,eps,num_batches=16,cuda=True):
        D=117
        H=100
        super(Eps_Greedy,self).__init__( bandit,num_batches)
        self.eps=eps
        self.cuda=cuda
        self.net=nn.Sequential(nn.Linear(D,H),
                               nn.ReLU(),
                               nn.Linear(H,H),
                                nn.ReLU(),
                               nn.Linear(H,2))
        if cuda: self.net=self.net.to(device='cuda')
        self.log=[]
        self.optimizer=torch.optim.SGD(self.net.parameters(),lr=1e-2)
    
    def get_action(self,ctx):
        p=torch.rand(1)
        
        if p<=self.eps:
            return choice([0,1])
        with torch.no_grad():
            if self.cuda: ctx=ctx.cuda()
            return self.net(ctx).argmax()
            
    
    def update(self):
        #if self.cuda: c,a,r=c.cuda(),a.cuda(),r.cuda()
        dl=torch.utils.data.DataLoader(self.buffer,64,True)
        for c,a,r in dl:
            pred_r=self.net(c)[torch.arange(len(c)),a]
            loss=((pred_r.reshape(-1)-r)**2).mean()
            loss.backward()
            self.log.append(float(loss.item()))
            self.optimizer.step()
            self.optimizer.zero_grad()
class Bayesian_Agent(Agent):
    def __init__(self,bandit,cuda=True,num_samples=2,beta=1e-3,lr=1e-4,num_epochs=10,rescale_kl=False,num_batches=16):
        D=117
        H=100
        super(Bayesian_Agent,self).__init__(bandit,num_batches)
        self.cuda=cuda
        self.num_samples=num_samples
        self.rescale_kl=rescale_kl
        self.num_batches=num_batches
        self.beta=beta
        self.net=nn.Sequential(BayesLinear(D,H),
                               nn.ReLU(),
                               BayesLinear(H,H),
                                nn.ReLU(),
                               BayesLinear(H,2))
        if cuda: self.net=self.net.to(device='cuda')
        self.optimizer=torch.optim.SGD(self.net.parameters(),lr=lr)
        self.log=[]
    
    def get_action(self,ctx):
        if self.cuda: ctx=ctx.cuda()
        mean_logits=0.0
        with torch.no_grad():
            for _ in range(2):
                mean_logits+=self.net(ctx)

        return mean_logits.argmax()
            
    
    def update(self):
        #if self.cuda: c,a,r=c.cuda(),a.cuda(),r.cuda()
        dl=torch.utils.data.DataLoader(self.buffer,64,True)
        for i,(c,a,r )in enumerate(dl):
            reg_losses=torch.zeros(self.num_samples)
            kl_losses=torch.zeros(self.num_samples)
            for i in range(self.num_samples):
                pred_r=self.net(c)[torch.arange(len(a)),a.reshape(-1)]
                kl_losses[i]=get_kl_loss(self.net, 1)
                reg_losses[i]=((pred_r.reshape(-1)-r)**2).mean()
            kl_loss,reg_loss= kl_losses.mean(),reg_losses.mean()
                    
            beta_= 2**(self.num_batches-i)/(2**self.num_batches)* self.beta if self.rescale_kl else self.beta
            loss=self.beta*kl_loss+reg_loss

            loss.backward()
            self.log.append((kl_loss.item(),reg_loss.item()))
            self.optimizer.step()
            self.optimizer.zero_grad()

if __name__=='__main__':
    bandits=[Mushroom_Bandit() for _ in range(4)]
    #agent=Eps_Greedy(bandit,0.01)
    agents=[Bayesian_Agent(bandits[0],beta=1e-8,lr=1e-2,num_batches=16),
            Eps_Greedy(bandits[0],0.0),
            Eps_Greedy(bandits[1],0.01),
            Eps_Greedy(bandits[2],0.05)]

    from tqdm import tqdm
    for agent,b,tag in zip(agents,bandits,['greedy','1pc_greedy','5pc_greedy']):
        for i in tqdm(range(40000),leave=True):
            agent.step()
            #if i%500==0 and i>=500: print(f'cum_reg {i} ',(b.cum_reg[-1]-b.cum_reg[-100])/100)
        np.save(f'/content/drive/MyDrive/colab_data/{tag}',(b.cum_reg,agent.log))
        plt.plot(b.cum_reg,label=tag)b


    plt.legend()
    
            
    
    
        
        
    
