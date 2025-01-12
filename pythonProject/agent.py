import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage.filters import threshold_otsu
from skimage.transform import resize

from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk

import random
import torch
from skimage.util import img_as_ubyte
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os
from skimage.color import rgb2gray


DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"

IMAGE_SAVE_DIR = "saved_images"

os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {})

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')

    def run(self, is_training=True, render=True):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training started..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode="human" if render else None, use_lidar=False)
        num_states = 64*32
        num_actions = 2

        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:

            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            step_count = 0
            epsilon_history = []
            best_reward = -99999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        BASE_IMAGE_SAVE_DIR = "images_per_episode"
        os.makedirs(BASE_IMAGE_SAVE_DIR, exist_ok=True)

        for episode in range(1000):
            index = 0


            if episode % 100 == 0:
                print(episode)
            state, _ = env.reset()

            image = env.render()


            # image_gray = rgb2gray(image)  # Convert to grayscale
            # # print(image_gray.shape)
            # image_gray_resized = resize(image_gray, (64, 32), anti_aliasing=True)
            # image_gray_resized = image_gray_resized / 255.0
            image_gray_resized = self.preprocess_image(image)
            # image_resized = np.resize(image_gray_resized, (288, 512))  # Example size, adjust as needed

            state = torch.tensor(image_gray_resized.flatten(), dtype=torch.float32).to(device)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                index = index + 1
                # Choose action
                if is_training and random.random() < epsilon:
                    #print("is training")
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.float, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax().item()

                new_state, reward, terminated, truncated, info = env.step(action)

                #print(reward)

                image = env.render()

                image_gray_resized = self.preprocess_image(image)
                episode_reward += reward

                new_state = torch.tensor(image_gray_resized.flatten(), dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append(
                        (state, torch.tensor(action, dtype=torch.int64, device=device), new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)
            # print(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f'{datetime.now().strftime(DATE_FORMAT)}: new best reward {episode_reward:.2f} at step {episode}'
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()


                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sammple(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def preprocess_image(self, image):
        gray_image = rgb2gray(image)
        resized_image = resize(gray_image, (64, 32), anti_aliasing=True)
        resized_image = resized_image / 255.0
        return resized_image.astype(np.float32)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Facultate RL')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set = args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False)
