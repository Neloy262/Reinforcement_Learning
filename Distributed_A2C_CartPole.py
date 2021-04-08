import torch as T
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import gym
import multiprocessing as mp
env = gym.make("CartPole-v0")
print(env.observation_space)


class Network(nn.Module):
    def __init__(self,lr,input_dims,actions,l1_dim,l2_dim):
        super(Network,self).__init__()
        self.lr=lr
        self.input_dims=input_dims
        self.actions=actions
        self.l1=nn.Linear(self.input_dims,8)
        self.l2=nn.Linear(8,8)
        self.l3=nn.Linear(8,actions)
        self.optimizer=optim.Adam(self.parameters(),lr=self.lr)
        self.device=T.device('cpu:0')
        self.to(self.device)
        
    def forward(self,observation):
        state=T.Tensor(observation).to(self.device)
        x=F.relu(self.l1(state))
        x=F.relu(self.l2(x))
        x=self.l3(x)
        return x

class Agent():
    def __init__(self,alpha,beta,input_dims,actions,l1_dim,l2_dim,gamma):
        self.gamma=gamma
        self.log_probs=None
        self.actor=Network(alpha,input_dims,actions,l1_dim,l2_dim).share_memory()
        self.critic=Network(beta,input_dims,1,l1_dim,l2_dim).share_memory()
    
    def act(self,observation):
        probs=F.softmax(self.actor.forward(observation))
        action_probs=T.distributions.Categorical(probs)
        action=action_probs.sample()
        self.log_probs=action_probs.log_prob(action)
        return action.item()
    
    def learn(self,state,reward,new_state,done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        q_val_state=self.critic.forward(state)
        q_val_new_state=self.critic.forward(new_state)
        
        delta=((reward+self.gamma*q_val_new_state*(1-int(done)))-q_val_state)
        
        actor_loss=-self.log_probs*delta
        critic_loss=delta**2
        
        (actor_loss+critic_loss).backward()
        
        self.actor.optimizer.step()
        self.critic.optimizer.step()

def worker(worker_agent,i,params):
    worker_env=gym.make("CartPole-v0")
    for i in range(params['epochs']):
        done=False
        score=0
        obs=worker_env.reset()
        while not done:
            action =worker_agent.act(obs)
            obs_, reward, done, info = worker_env.step(action)
            worker_agent.learn(obs,reward,obs_,done)
            obs=obs_
            score+=reward
        print("Score for worker:",i," ",score)

if __name__=='__main__':

	agent=Agent(0.0001,0.0001,4,2,32,32,0.95)
	process_list=[]
	params={
    		'epochs':1000,
    		'n_workers':7
	}

	for i in range(params['n_workers']):
    		p=mp.Process(target=worker, args=(agent,i,params))
    		p.start()
    		process_list.append(p)
    
	for p in process_list:
    		p.join()
    
	for p in process_list:
    		p.terminate()