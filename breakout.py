import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torchvision.transforms as transforms

# 设置超参数
GAMMA = 0.99  # 折扣因子
EPSILON_START = 1.0  # 探索率的初始值
EPSILON_DECAY = 0.995  # 探索率的衰减率
EPSILON_MIN = 0.01  # 探索率的最小值
MEMORY_SIZE = 10000  # 经验回放缓冲区的最大大小
BATCH_SIZE = 32  # 批次大小
TARGET_UPDATE = 10  # 更新目标网络的频率

# 定义 DQN 网络


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建环境
env = gym.make('Breakout-v0', render_mode='human')
w, h, c = env.observation_space.shape
action_size = env.action_space.n

print(w, h, c, action_size)
# 创建 DQN 网络和目标网络
model = DQN(w*h, action_size)
target_model = DQN(w*h, action_size)
target_model.load_state_dict(model.state_dict())
target_model.eval()

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 创建记忆回放缓冲区
memory = deque(maxlen=MEMORY_SIZE)

# 初始化探索率
epsilon = EPSILON_START

# 定义transforms，将张量转换为灰度图像
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# 训练循环
for episode in range(1000):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        # 选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()  # 随机探索
        else:
            state_tensor = torch.tensor(
                state, dtype=torch.float32)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()  # 基于当前策略选择动作

        # 执行动作并观察结果
        next_state, reward, done, _, _ = env.step(action)

        # 存储经验到记忆回放缓冲区
        # action_tensor = torch.nn.functional.one_hot(
        #     torch.tensor(action), action_size)
        memory.append((state), action,
                      reward, transform(next_state), done))

        state = next_state
        total_reward += reward

        # 执行一次训练
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(
                *batch)

            # print(len(batch_states))
            # print(type(batch_actions))
            # print(batch_actions[0].shape)

            # 32*100800
            batch_states = torch.tensor(
                batch_states, dtype=torch.float32)
            # 1*1*32
            batch_actions = torch.tensor(batch_actions).unsqueeze(1)
            # 32*1
            batch_rewards = torch.tensor(
                batch_rewards, dtype=torch.float32).unsqueeze(1)
            # 32*100800
            batch_next_states = torch.tensor(
                batch_states, dtype=torch.float32)
            # 32*1
            batch_dones = torch.tensor(
                batch_dones, dtype=torch.float32).unsqueeze(1)

            print(batch_states.shape, batch_actions.shape,
                  batch_rewards.shape, batch_next_states.shape, batch_dones.shape)

            # 32*4
            q_values = model(batch_states)
            predicted_q_values = q_values.gather(1, batch_actions)
            # print("predicted_q_values shape:", predicted_q_values.shape)
            # print(q_values, predicted_q_values)

            with torch.no_grad():
                # 32*4
                next_q_values = target_model(batch_next_states)
                # 32*1
                max_next_q_values = torch.max(
                    next_q_values, dim=1, keepdim=True)[0]
                # 32*1
                target_q_values = batch_rewards + \
                    (1 - batch_dones) * GAMMA * next_q_values

            # print("target_q_values:", target_q_values.shape)
            loss = criterion(predicted_q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 更新目标网络
    if episode % TARGET_UPDATE == 0:
        target_model.load_state_dict(model.state_dict())

    # 衰减探索率
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    print("Episode: {}, Total Reward: {}, Epsilon: {:.4f}".format(
        episode, total_reward, epsilon))
