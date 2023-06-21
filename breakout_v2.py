import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Step 1: Environment Setup
env = gym.make('Breakout-v0', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(state_dim, action_dim)
# Step 2: Define the Q-Network


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255.0  # Normalize input to a range of [0, 1]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: Experience Replay Buffer


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Step 4: Epsilon-Greedy Exploration


def epsilon_greedy_policy(q_values, epsilon):
    # Implement the epsilon-greedy policy
    if random.random() < epsilon:
        # Choose a random action (exploration)
        action = random.randint(0, q_values.shape[0] - 1)
    else:
        # Choose the action with the highest Q-value (exploitation)
        action = torch.argmax(q_values).item()
    return action

# Step 5: DQN Algorithm


def DQN(env, q_network, target_network, replay_buffer, num_episodes, batch_size, gamma, epsilon, epsilon_decay, update_target_freq):
    # Initialize the target network with the weights of the Q-network
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            # Select action using epsilon-greedy policy
            state_tensor = torch.tensor(state).unsqueeze(0).permute(0, 3, 1, 2)
            print("state_tensor shape:", state_tensor.shape)
            q_values = q_network(state_tensor.float())

            # Take action in the environment and observe the next state and reward
            action = epsilon_greedy_policy(q_values, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            # Store the transition in the replay buffer
            replay_buffer.add(state, action, reward, next, done)

            # Sample a batch of transitions from the replay buffer
            if len(replay_buffer) < batch_size:
             # Sample a batch of transitions from the replay buffer
                batch = replay_buffer.sample(batch_size)

                # Separate the batch into individual components
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert the batch elements into tensors
                # 32*240
                states = torch.tensor(states).float()
                # 32*1
                actions = torch.tensor(actions)
                # 32*1
                rewards = torch.tensor(rewards).float()
                # 32*240
                next_states = torch.tensor(next_states).float()
                # 32*1
                dones = torch.tensor(dones)

                # Compute Q-values for current and next states
                q_values = q_network(states)
                next_q_values = target_network(next_states)

                # Compute the target Q-values
                target_q_values = q_values.clone()
                target_q_values[np.arange(batch_size), actions] = rewards + \
                    gamma * torch.max(next_q_values,
                                      dim=1).values * (1 - dones)

                # Compute the loss and perform a gradient descent step
                loss = criterion(q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update the current state and cumulative reward

            state = next_state

            if done:
                break

            # Update the target network every update_target_freq steps
            if episode % update_target_freq == 0:
                target_network.load_state_dict(q_network.state_dict())

            print(f"Episode {episode+1}: Reward = {episode_reward}")
            epsilon = max(epsilon * epsilon_decay, 0.01)

            return q_network


# Step 6: Training
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
replay_buffer = ReplayBuffer(capacity=10000)
num_episodes = 1000
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.99
update_target_freq = 100

trained_q_network = DQN(env, q_network, target_network, replay_buffer,
                        num_episodes, batch_size, gamma, epsilon, epsilon_decay, update_target_freq)

# # Step 7: Testing
# epsilon = 0.0  # Disable exploration during testing
# total_reward = 0

# for episode in range(10):  # Run 10 test episodes
#     state = env.reset()

#     while True:
#         # Select action using epsilon-greedy policy (without exploration)

#         # Take action in the environment and observe the next state and reward

#         # Update the current state and cumulative reward

#         if done:
#             break

#     print("Episode {}: Total Reward = {}".format(episode + 1, total_reward))
