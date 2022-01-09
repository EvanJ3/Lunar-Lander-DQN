import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Q_Network(nn.Module):
    '''
    Creates a pytorch dynamically compiled Q Network.
    
    '''
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32,32),
                 activation=F.relu):
        
        super(Q_Network,self).__init__()
        
        self.activation_fn = activation
        self.input_layer = nn.Linear(input_dim,hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        
        for i in range(0,len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1],output_dim)

    def _format(self,state):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,device=self.device,dtype=torch.float32)
            x = x.unsqueeze(0)
        return x
    
    def forward(self,state):
        x = self._format(state)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)
        return x
    
    def load_data(self,states,actions,rewards,next_states,dones):
        
        states = torch.tensor(states,device=self.device,dtype=torch.float32)
        actions = torch.tensor(actions,device=self.device,dtype=torch.long)
        next_states = torch.tensor(next_states,device=self.device,dtype=torch.float32)
        rewards = torch.tensor(rewards,device=self.device,dtype=torch.float32)
        dones = torch.tensor(dones,device=self.device,dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    
    
    

class Avg_Dueling_Q_Network(nn.Module):
    '''
    Creates a pytorch dynamically compiled average q value based dueling Q Network.
    
    '''
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32,32),
                 activation=F.relu):
        
        super(Avg_Dueling_Q_Network,self).__init__()
        
        self.activation_fn = activation
        self.input_layer = nn.Linear(input_dim,hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        
        for i in range(0,len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            
        self.value_output = nn.Linear(hidden_dims[-1],1)
        self.advantage_output = nn.Linear(hidden_dims[-1],output_dim)
        
        
    def _format(self,state):
        x = state
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,device=self.device,dtype=torch.float32)
            x = x.unsqueeze(0)
        return x
    
    def forward(self,state):
        x = self._format(state)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        a = self.advantage_output(x)
        v = self.value_output(x)
        v = v.expand_as(a)
        q = v + a - a.mean(1,keepdim=True).expand_as(a)
        return q
    
    def load_data(self,states,actions,rewards,next_states,dones):

        states = torch.tensor(states,device=self.device,dtype=torch.float32)
        actions = torch.tensor(actions,device=self.device,dtype=torch.long)
        next_states = torch.tensor(next_states,device=self.device,dtype=torch.float32)
        rewards = torch.tensor(rewards,device=self.device,dtype=torch.float32)
        dones = torch.tensor(dones,device=self.device,dtype=torch.float32)
        return states, actions, rewards, next_states, dones