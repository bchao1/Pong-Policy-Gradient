import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from scipy.misc import imsave, imresize
from model import Actor


def preprocess_frame(frame):
    frame = 0.2126 * frame[:,:,0] + 0.7152 * frame[:,:,1] + 0.0722 * frame[:,:,2]
    frame = frame.astype(np.uint8)
    frame = imresize(frame, (80, 80))
    frame = frame.ravel()
    return torch.tensor(frame, dtype = torch.float).unsqueeze(0) # Return a tensor

def get_action_label(prob):
    prob = prob.detach().cpu().numpy()
    action = 2 if prob[0] > np.random.uniform() else 3 # 2 is up / 3 is down
    return action

# NN actor config
actor = Actor
actor.load_state_dict(torch.load('actor.ckpt', map_location = 'cpu'))

# Environment config
env = gym.make("Pong-v0")
input_frame = env.reset()
prev_frame = None
reward_sum = 0
episode_number = 1

while True:
    env.render()
    curr_frame = preprocess_frame(input_frame)
    frame_diff = curr_frame - (prev_frame if prev_frame is not None else 0)
    prev_frame = curr_frame
    
    frame_diff = frame_diff
    prob = actor(frame_diff).view(-1) #1D probability: moving up
    action,= get_action_label(prob) #Supervised hard label
    
    input_frame, reward, done, info = env.step(action) #Make a move according to the action
    
    reward_sum += reward #Accumulate reward of one episode
    if done:
        episode_number += 1
        print("Episode: {}, Reward: {}".format(episode_number, reward_sum))
        
        # Resetting stuff
        reward_sum = 0 #Reset episode reward sum
        input_frame = env.reset() #Reset environment
        prev_frame = None