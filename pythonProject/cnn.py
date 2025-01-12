import gymnasium as gym
import numpy as np

import torch.nn.functional as F


from skimage.transform import resize


import random
import torch
from torch import nn, optim
import yaml

from collections import deque


from datetime import datetime, timedelta


import flappy_bird_gymnasium
import os
from skimage.color import rgb2gray

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ReplayMemory:
    def __init__(self, maxlen, seed=None):
        self.memory = deque(maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training started..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode="human" if render else None, use_lidar=False)
        num_actions = 2

        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_actions).to(device)

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            step_count = 0
            epsilon_history = []
            best_reward = -99999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in range(55000):
            index = 0
            if episode % 1000 == 0:
                print(episode)
            state, _ = env.reset()
            image = env.render()

            state = self.preprocess_image(image)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.float, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state).argmax().item()

                new_state, reward, terminated, truncated, info = env.step(action)

                image = env.render()
                new_state = self.preprocess_image(image)
                episode_reward += reward

                if is_training:
                    memory.append(
                        (state, torch.tensor(action, dtype=torch.int64, device=device), new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f'{datetime.now().strftime(DATE_FORMAT)}: new best reward {episode_reward:.2f} at step {episode}'
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def preprocess_image(self, image):
        image_resized = resize(image, (84, 84), anti_aliasing=True)
        image_resized = np.transpose(image_resized, (2, 0, 1))
        return torch.tensor(image_resized, dtype=torch.float32).to(device)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
      states, actions, new_states, rewards, terminations = zip(*mini_batch)

      states = torch.stack(states).to(device)
      actions = torch.stack(actions).to(device)
      new_states = torch.stack(new_states).to(device)
      rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
      terminations = torch.tensor(terminations, dtype=torch.float32, device=device)

      with torch.no_grad():
          target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

      current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

      loss = self.loss_fn(current_q, target_q)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()


if __name__ == '__main__':
    hyperparameter_set = 'flappybird1'
    dql = Agent(hyperparameter_set=hyperparameter_set)
    dql.run(is_training=False)