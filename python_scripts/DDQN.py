import sys
import time
import os
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PyTorch_Networks import Q_Network,Avg_Dueling_Q_Network
from PyTorch_TB_Helper import clean_tensorboard, create_tensor_board_callback
from Action_Strategies import E_Greedy_Exp

clean_tensorboard()
%load_ext tensorboard
%tensorboard --logdir 'C:\Users\Eaj59\Documents\RL_Projects\Project_2_DRL\Pytorch_Models\log_dir'


class Agent():
    
    def __init__(self,env_name,
                 hidden_dims,
                 action_fn,
                 activation_fn=F.relu,
                 gamma=.99,
                 lr=0.001,
                 tau=0.01,
                 capacity=20000,
                 batch_size=32,
                 enable_dueling=False,
                 dueling_type='avg',
                 enable_TB=False,
                 model_name='Model',
                 seed=None,
                 keep_recents=100):
        
        self.env = gym.make(env_name)
        self.seed = seed
        
        if self.seed is not None:
            torch.manual_seed(self.seed) 
            np.random.seed(self.seed)
            random.seed(self.seed)
            self.env.seed(self.seed)
        
        self.env_state = self.env.reset()
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.shape[0]
        
        self.enable_dueling = enable_dueling
        self.activation_fn = activation_fn
        self.dueling_type = dueling_type
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.action_fn = action_fn
        self.capacity = capacity
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        
        
        self.recent_avg_rewards = deque(maxlen=keep_recents)
        self.episode_reward = 0
        self.episode_counter = 0
        self.episode_time_step_counter = 0
       
    
        self.online = self.make_online()
        self.target = self.make_online()
        self.value_optimizer = optim.RMSprop(self.online.parameters(),lr=self.lr)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        
        self.buffer = ReplayBuffer(self.capacity)
        
        self.enable_TB = enable_TB
        self.model_name = model_name
        self.run_id = None
        self.writer = None
        
        
        if self.enable_TB:
            self.run_id,self.writer = create_tensor_board_callback(model_name=self.model_name)
            with self.writer as w:
                w.add_graph(self.online,torch.rand(self.batch_size,self.n_states).detach().to('cuda:0'))
                w.close()
    
    
    def write_TB_episode(self):
        with self.writer as w:
            w.add_scalar('episodic_reward',self.episode_reward,global_step=self.episode_counter,walltime=time.time())
            w.add_scalar('epsilon',self.action_fn.current_epsilon,global_step=self.episode_counter,walltime=time.time())
            w.add_scalar('Average Last 100 Rewards',np.mean(self.recent_avg_rewards),global_step=self.episode_counter,walltime=time.time())
            w.close()
            
    

    def update_seed(self,custom_seed=None):
        if custom_seed is None:
            self.seed += 1
            self.env.seed(self.seed)
            torch.manual_seed(self.seed) 
            np.random.seed(self.seed) 
            random.seed(self.seed)
        else:    
            self.seed = custom_seed
            self.env.seed(self.seed)
            torch.manual_seed(self.seed) 
            np.random.seed(self.seed) 
            random.seed(self.seed)

    def make_online(self):
        if self.enable_dueling:
            
            if self.dueling_type == 'avg':
                return Avg_Dueling_Q_Network(input_dim=self.n_states,output_dim=self.n_actions,hidden_dims=self.hidden_dims,activation=self.activation_fn)
            else:
                return Max_Dueling_Q_Network(input_dim=self.n_states,output_dim=self.n_actions,hidden_dims=self.hidden_dims,activation=self.activation_fn)
        
        else:
            return Q_Network(input_dim=self.n_states,output_dim=self.n_actions,hidden_dims=self.hidden_dims,activation=self.activation_fn)

        
    def reset_env(self):
        self.recent_avg_rewards.append(self.episode_reward)
        self.episode_counter += 1
        
        if self.seed is not None:
            self.update_seed()
      
        if self.enable_TB:
            self.write_TB_episode()
     
        self.episode_time_step_counter = 0
        self.episode_reward = 0
        self.env_state = self.env.reset()

    def play_one_step(self):
        state = np.array(self.env_state)
        action = self.action_fn.Select_Action(self.online,state,self.n_actions)
        next_state, reward, done, info = self.env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = done and not is_truncated
        self.buffer.add_exp(state=state, action=action, reward=reward, next_state=next_state, done=float(is_failure))
        self.env_state = next_state
        self.episode_reward += reward
        self.episode_time_step_counter += 1
        return done
    
         
    def optimizer_step(self,batch_size_instance,states,actions,rewards,next_states,dones):
        
        max_online_prime_indices = self.online(next_states).detach().max(1)[1]
        target_q_vals = self.target(next_states).detach()
        max_target_q_vals = target_q_vals[np.arange(batch_size_instance),max_online_prime_indices]
        max_target_q_vals = max_target_q_vals.unsqueeze(1)
        updated_targets = rewards + self.gamma * (1-dones) * max_target_q_vals
        current_preds = self.online(states).gather(1,actions.unsqueeze(1))
        td_error = current_preds - updated_targets
        loss_value = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(),float('inf'))
        self.value_optimizer.step()


    def training_step(self):
        if self.buffer.__len__() < self.batch_size:
            batch_size_instance = self.buffer.__len__()
            states,actions,rewards,next_states,dones = self.buffer.exp_sample(batch_size=batch_size_instance)
        else:
            batch_size_instance = self.batch_size
            states,actions,rewards,next_states,dones = self.buffer.exp_sample(batch_size=self.batch_size)
    
        states,actions,rewards,next_states,dones = self.online.load_data(states,actions,rewards,next_states,dones)
        self.optimizer_step(batch_size_instance,states,actions,rewards,next_states,dones)
        
    def play_warmup_batches(self,num_warmup_batches=5):
        for i in range(num_warmup_batches):
            self.env_state = self.env.reset()
            done = False
            while not(done):
                state = np.array(self.env_state)
                action = self.action_fn.Select_Action(self.online,state,self.n_actions)
                next_state, reward, done, info = self.env.step(action)
                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_failure = done and not is_truncated
                self.buffer.add_exp(state=state, action=action, reward=reward, next_state=next_state, done=float(is_failure))
                self.env_state = next_state
            if self.seed is not None:
                self.update_seed()
        self.action_fn.Reset_Epsilon_Schedule()
                

    
    def update_target_network(self):
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    
    def soft_target_update(self):
        for target,online in zip(self.target.parameters(),self.online.parameters()):
            target.data.copy_((1.0 - self.tau) * target.data + self.tau * online.data)
            
    def training_loop(self,max_episodes):
        for i in range(max_episodes):
            self.reset_env()
            done = False
            while not(done):
                done = self.play_one_step()
                self.training_step()
            self.update_target_network()
            
            
            
class ReplayBuffer():
    
    def __init__(self,max_len):
        self.max_len = max_len
        self.buffer = deque(maxlen=self.max_len)
        
    
    def add_exp(self,state,action,reward,next_state,done):
        exp = (state,action,reward, next_state, done)
        if len(self.buffer)<= self.max_len:
            self.buffer.append(exp)
        else:
            self.buffer[0] = exp
        
    def __len__(self):
        return len(self.buffer)
    
    def exp_sample(self,batch_size):
        indices = np.random.randint(len(self.buffer), size=batch_size)
        batch = [self.buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [np.array([experience[entry] for experience in batch])for entry in range(5)]
        return states, actions, rewards[:,np.newaxis], next_states, dones[:,np.newaxis]
             
        
        
my_agent = Agent('LunarLander-v2',hidden_dims=(128,64,32),enable_TB=True,action_fn=E_Greedy_Exp(1,0.075,100000),model_name='Lunar_Lander_LR_0015_Run_1',capacity=20000,lr=0.0015)
max_episodes = 5000

my_agent.play_warmup_batches(num_warmup_batches=10)
start_time = time.time()
for i in range(0,max_episodes):
    my_agent.reset_env()
    done = False
    while not(done):
        done = my_agent.play_one_step()
        my_agent.training_step()

    my_agent.update_target_network()
    if np.mean(my_agent.recent_avg_rewards) >= 200.00:
        end_time = time.time()
        time_diff = end_time-start_time
        print(f'DDQN Terminated at Step {i} with wall clock time of {time_diff}')
        break
        
path_model = os.path.join(os.getcwd(),my_agent.model_name)
print(path_model)
torch.save(my_agent.online,path_model)