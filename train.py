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
parser.add_argument("--render_train", default=False, type=bool, help="Render the training steps (default: False)")
parser.add_argument("--render_eval", default=False, type=bool, help="Render the evaluation steps (default: False)")
parser.add_argument("--load_model", default=False, type=bool, help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default="./saved_models/", help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--seed", default=0, type=int, help="Random seed (default: 0)")
parser.add_argument("--episodes", default=1e6, type=int, help="Num. of total timesteps of training (default: 1e6)")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size (default: 64; OpenAI: 128)")
parser.add_argument("--replay_size", default=1e6, type=int, help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
parser.add_argument("--gamma", default=0.99, help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001, help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--noise_stddev", default=0.2, type=int, help="Standard deviation of the OU-Noise (default: 0.2)")
parser.add_argument("--hidden_size", nargs=2, default=[256,256,128,128,64], type=tuple, help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")
parser.add_argument("--n_test_cycles", default=30, type=int, help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
args = parser.parse_args()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.makedirs(f"./policy_{args.env}/",exist_ok=True)
if __name__ == "__main__":
    checkpoint_dir = args.save_dir + args.env
    kwargs = dict(exclude_current_positions_from_observation=False)
    env = gym.make(f"{args.env}")
    # env = NormalizedActions(env)
    # reward_threshold = gym.spec(args.env).reward_threshold if gym.spec(args.env).reward_threshold is not None else np.inf
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    hidden_size = tuple(args.hidden_size)
    agent = DDPG(gamma=args.gamma, tau=args.tau, hidden_size=hidden_size, num_inputs=env.observation_space.shape[0], action_space=env.action_space, checkpoint_dir=checkpoint_dir)
    memory = ReplayMemory(int(args.replay_size))

    # Initialize OU-Noise
    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(args.noise_stddev) * np.ones(nb_actions))
    start_step = 0
    time_last_checkpoint = time.time()
    best_policy_test_reward = -1000000000
    for episode in range(int(args.episodes)):
        ou_noise.reset()
        episode_return = 0
        state = torch.Tensor([env.reset()[0]]).to(device)
        while True:
            if args.render_train:
                env.render()
            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _,info = env.step(action.cpu().numpy()[0])
            episode_return += reward
            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)
            memory.push(state, action, mask, next_state, reward)
            state = next_state
            epoch_value_loss = 0
            epoch_policy_loss = 0
            if len(memory) > args.batch_size:
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss
            if done or _:
                break
        test_rewards = []
        for test_count in range(5):
            state = torch.Tensor([env.reset()[0]]).to(device)
            test_reward = 0
            while True:
                if args.render_eval:
                    env.render()
                action = agent.calc_action(state)  # Selection without noise
                next_state, reward, done, _,info = env.step(action.cpu().numpy()[0])
                test_reward += reward
                next_state = torch.Tensor([next_state]).to(device)
                state = next_state
                if test_reward % 10000 ==0:
                    print(f'episode:{episode} test:{test_count} test reward: {test_reward}')
                    torch.save(agent.actor.state_dict(),f"./policy_{args.env}/episode_{str(episode).zfill(5)}_test_reward_{test_reward}.pth")
                if done or _:
                    break
            test_rewards.append(test_reward)
        print(f"episode {episode} mean:{np.mean(test_rewards)} best: {best_policy_test_reward}")
        if np.mean(test_rewards)>best_policy_test_reward:
            best_policy_test_reward = np.mean(test_rewards)
            torch.save(agent.actor.state_dict(),f"./policy_{args.env}/episode_{str(episode).zfill(5)}_{best_policy_test_reward}.pth")
            torch.save(agent.actor.state_dict(),f"./policy_{args.env}/best.pth")
    env.close()