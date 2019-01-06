import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.misc import imsave, imresize
from model import Actor
'''
    Minimal implementation of Pong Policy Gradient.
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_weights(m):
    ''' Define own weight initialization.
    '''
    pass

def get_prepro_state(s):
    '''
    Args:
        s: state of current environment. In Pong, it is an 160 * 210 * 3 RGB frame.
    Returns:
        A flattened tensor of length 6400.
    '''
    s = 0.2126 * s[:, :, 0] + 0.7152 * s[:, :, 1] + 0.0722 * s[:, :, 2]
    s = s.astype(np.uint8)
    s = imresize(s, (80, 80)).ravel()
    return torch.tensor(s, dtype = torch.float).unsqueeze(0).to(device) # Return a tensor

def get_po_loss(p, l, r):
    '''
    Args:
        p: list of probabilities of moving up in an episode.
        l: list of  hot labels indicating the action (up or down) in an epsiode (1 = up, 0 = down).
        r: list of rewards in an episode.
    Returns:
        The policy gradient loss.
    '''
    
    p = torch.cat(p, 0)
    l = torch.cat(l, 0)
    r = get_discounted_reward(np.array(r), gamma)
    
    eps = 1e-8
    return -torch.sum((l * torch.log(p + eps) + (1 - l) * torch.log(1 - p + eps)) * r)

def get_discounted_reward(r, gamma):
    '''
    Args:
        r: a list of rewards in an episode.
        gamma: discount factor.
    Returns:
        A dicounted and normalized reward list (tensor).
    '''
    dsct_r = np.zeros_like(r)
    run_r = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 
        run_r = run_r * gamma + r[t]
        dsct_r[t] = run_r
    dsct_r = (dsct_r - np.mean(dsct_r)) / np.std(dsct_r)
    return torch.tensor(dsct_r, dtype = torch.float).to(device)

def get_action_label(p):
    ''' Get action based on predicted probability
    Args:
        p: The probability to move up.
    Returns:
        action: 2 = up / 3 = down.
        label: 1 = up / 0 = down.
    '''
    p = p.detach().cpu().numpy()
    action = 2 if p[0] > np.random.uniform() else 3 # 2 is up / 3 is down
    label =  torch.ones(1) if action == 2 else torch.zeros(1) # Label is 1 if moving up
    return action, label.to(device)

def get
resume = False

# NN config
actor = Actor.to(device)
if resume:
    actor.load_state_dict(torch.load('actor.ckpt'))

optimizer = optim.Adam(actor.parameters(), lr = 0.0001)
optimizer.zero_grad()

# Environment config
env = gym.make("Pong-v0")
prev_state, curr_state = env.reset(), None

# Some tracking variables
reward_sum = 0
episode_number = 0
batch_size = 10
gamma = 0.99
reward_history, reward_list, prob_list, label_list = [], [], [], []

while True:
    curr_state = get_prepro_state(curr_state).to(device)
    diff_state = curr_state - (prev_state if prev_state is not None else 0)
    prev_state = curr_state

    prob = actor(diff_state).view(-1) # 1D probability: moving up
    action, label = get_action_label(prob) # Supervised hard label
    label_list.append(label)
    prob_list.append(prob)
    
    curr_state, reward, done, info = env.step(action) #Make a move according to the action
    
    reward_sum += reward #Accumulate reward of one episode
    reward_list.append(reward) #A previous list of rewards
    
    if done:
        episode_number += 1
        reward_history.append(reward_sum)
        np.save('reward.npy', np.array(reward_history))
        
        loss = get_po_loss(prob_list, label_list, reward_list)
        loss.backward() # Accumulate gradients
            
        print("Episode: {}, Reward: {}, Loss: {}".format(episode_number, reward_sum, loss.item()))
        
        if episode_number % batch_size == 0:
            optimizer.step() # Make nn update
            optimizer.zero_grad() # Clear gradients
            
        if episode_number % 100 == 0:
            torch.save(actor.state_dict(), 'actor.ckpt')
        
        # Resetting stuff
        reward_sum = 0
        prob_list, label_list, reward_list = [], [], []
        curr_state, prev_state = env.reset(), None