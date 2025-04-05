import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torchvision.transforms as T
from transformers import Trainer,DataCollatorForLanguageModeling

# 定义策略网络


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._get_conv_output(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((240, 160)),
            T.ToTensor()
        ])

    def _get_conv_output(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        x = self.transform(x)
        x = x.float() / 255.0  # 归一化像素值
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 将卷积层输出的特征图扁平化成一维向量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# 定义PPO算法


class PPO:
    def __init__(self, input_shape, num_actions):
        self.policy = PolicyNetwork(input_shape, num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 0.2

    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update(self, states, actions, log_probs, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        advantages = torch.FloatTensor(advantages)

        old_probs = torch.exp(log_probs.detach())

        for _ in range(10):  # 进行多次优化迭代
            probs = self.policy(states)
            m = Categorical(probs)
            entropy = m.entropy()

            ratios = torch.exp(m.log_prob(actions) - log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon,
                                1 + self.epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.smooth_l1_loss(
                self.policy(states), old_probs.unsqueeze(1))
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# 创建Breakout环境
env = gym.make('Breakout-v0')
state_shape = env.observation_space.shape
num_actions = env.action_space.n

print(state_shape, num_actions)
# 初始化PPO代理
ppo_agent = PPO(state_shape, num_actions)

# 训练PPO代理
num_epochs = 1000
max_timesteps = 1000
for epoch in range(num_epochs):
    state = env.reset()
    done = False
    total_reward = 0
    timesteps = 0

    while not done and timesteps < max_timesteps:
        action, log_prob = ppo_agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)

        ppo_agent.update([state], [action], [log_prob], [reward])

        state = next_state
        total_reward += reward
        timesteps += 1

    print(f"Epoch {epoch+1}: Total Reward = {total_reward}")
