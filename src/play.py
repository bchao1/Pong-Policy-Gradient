import torch
import gym
import numpy as np
from scipy.misc import imsave, imresize
from model import Actor
import subprocess

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def reset():
    subprocess.call("rm play.mp4".split(' '))
    subprocess.call("rm -rf frames".split(' '))
    subprocess.call("mkdir frames".split(' '))
    return 0, 0, 0, 1, False

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


actor = Actor.to(device)
actor.load_state_dict(torch.load('actor.ckpt'))

env = gym.make("Pong-v0")
prev_frame, curr_frame = None, env.reset()
opponent_score, your_score, total_reward, frame_no, done = reset()

print("   << Playing Pong >>   ")
print("=" * 24)
while not done:
    curr_frame = preprocess_frame(curr_frame)
    frame_diff = curr_frame - (prev_frame if prev_frame is not None else 0)
    prev_frame = curr_frame
    
    frame_diff = frame_diff.to(device)
    prob = actor(frame_diff).view(-1) 
    action = get_action_label(prob) 
    
    curr_frame, reward, done, info = env.step(action) 
    
    if reward == 1: your_score += 1
    elif reward == -1: opponent_score += 1
    if reward != 0:
        print("Opponent [{}] | You [{}]".format(str(opponent_score).rjust(2), str(your_score).rjust(2)), end = "\r")
        
    imsave('./frames/{}.png'.format(str(frame_no).zfill(10)), curr_frame)
    frame_no += 1
    total_reward += reward 

print("\nReward: {}".format(total_reward))
subprocess.call("ffmpeg -framerate 128 -i ./frames/%010d.png -crf 0 play.mp4".split(" "))
