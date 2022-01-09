import numpy as np
import torch
import math


class Pure_Random():
    '''
    As the name implies this is a uniformly random action selection function used to simulated randomized play against an agent 
    or other benchmark. Returns random actions 
    '''
    
    def __init__(self):
        
        pass

    def Select_Action(self,n_actions):
        return np.random.randint(n_actions)
    
    
class E_Greedy():
    '''
    Regualar E greedy selection with no decay this is a constant epsilon version values are never decayed.
    
    '''
    
    
    def __init__(self,epsilon=.5):
        self.current_epsilon = epsilon
        

    def update_epsilon(self,new_epsilon):
        self.current_epsilon = new_epsilon
        
        
    @torch.no_grad()
    def Select_Action(self,model,state,n_actions):
        if np.random.rand() < self.current_epsilon:
            return np.random.randint(n_actions)
        else:
            return model(state).to('cpu').max(1)[1].numpy()[0]
        
        

    
class E_Greedy_Exp():
    '''
    Exponentially decayed E greedy epsilon selection from starting value of epsilon to minimum value of epsilon within
    the specified step range. Defaults to min epsilon outside of max steps.
    
    '''
    
    def __init__(self,initial_epsilon=1.0,min_epsilon=0.1,num_steps=20000):
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.num_steps = num_steps
        self.current_epsilon = initial_epsilon
        self.current_step = 0
        self.epsilons = np.logspace(-2, 0, num_steps, endpoint=True)
        self.epsilons = self.epsilons * (initial_epsilon - min_epsilon) + min_epsilon
    
   
    def update_epsilon(self):
        self.current_epsilon = self.min_epsilon if self.current_step >= self.num_steps else self.epsilons[self.current_step]
        self.current_step += 1
            

    def Reset_Epsilon_Schedule(self):
        self.initial_epsilon = self.initial_epsilon
        self.min_epsilon = self.min_epsilon
        self.num_steps = self.num_steps
        self.current_epsilon = self.initial_epsilon
        self.current_step = 0
        self.epsilons =np.logspace(-2, 0, self.num_steps, endpoint=True)
        self.epsilons = self.epsilons * (self.initial_epsilon - self.min_epsilon) + self.min_epsilon
        
    @torch.no_grad()
    def Select_Action(self,model,state,n_actions):
        if np.random.rand() <= self.current_epsilon:
            self.update_epsilon()
            return np.random.randint(low=0,high=n_actions)
        else:
            self.update_epsilon()
            return model(state).to('cpu').max(1)[1].numpy()[0]

    
