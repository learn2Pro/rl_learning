import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gym

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.00025
MEMORY_SIZE = 1000000
BATCH_SIZE = 64
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.00001


class DQN(nn.Module):
    def __init__(self, state_shape, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.exploration_rate = EXPLORATION_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = DQN(state_shape, action_space).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_next = torch.tensor(
                state_next, dtype=torch.float32).unsqueeze(0)
            action = torch.tensor([action], dtype=torch.int64).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)

            q_values = self.model(state)
            q_values_next = self.model(state_next)
            q_update = reward
            if not terminal:
                q_update = reward + GAMMA * torch.max(q_values_next)
            q_values_target = q_values.clone()
            q_values_target[0, action] = q_update

            loss = self.loss_fn(q_values, q_values_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def preprocess_state(state):
    ans = np.expand_dims(state, axis=0)
    print(ans.shape)
    return ans


def main():
    env = gym.make('Breakout-v0')
    state_shape = (4, 84, 84)
    action_space = env.action_space.n
    agent = DQNAgent(state_shape, action_space)

    episodes = 1000
    for e in range(episodes):
        state = preprocess_state(env.reset()[0])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.experience_replay()

        print(
            f"Episode: {e + 1}, exploration rate: {agent.exploration_rate:.2f}")


if __name__ == "__main__":
    main()
