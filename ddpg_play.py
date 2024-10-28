import argparse
import logging
import os
import random
import time
import glob,natsort,shutil
import gym
import numpy as np
import torch,os
from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
from wrappers.normalized_actions import NormalizedActions
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Humanoid-v4", help="the environment on which the agent should be trained (Default: InvertedPendulum-v4)")
parser.add_argument("--seed", default=0, type=int, help="Random seed (default: 0)")
parser.add_argument("--hidden_size", nargs=2, default=[256,256,128,128,64], type=tuple, help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")
parser.add_argument("--gamma", default=0.99, help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001, help="Update factor for the soft update of the target networks (default: 0.001)")
args = parser.parse_args()
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    kwargs = dict()
    # env = gym.make(args.env, render_mode='human')
    env = gym.make(args.env)
    # Define and build DDPG agent
    hidden_size = tuple(args.hidden_size)
    agent = DDPG(gamma=args.gamma, tau=args.tau, hidden_size=hidden_size, num_inputs=env.observation_space.shape[0], action_space=env.action_space)
    agent.actor.load_state_dict(torch.load("./policy_Humanoid-v4/best.pth",map_location=torch.device(device)))
    
    state = torch.Tensor([env.reset()[0]]).to(device)
    episode_return = 0
    while True:
        action = agent.calc_action(state, action_noise=None)
        q_value = agent.critic(state, action)
        next_state, reward, done, _,info = env.step(action.cpu().numpy()[0])
        episode_return += reward
        state = torch.Tensor([next_state]).to(device)
        env.render()
        print(episode_return)
        if done:
            break
