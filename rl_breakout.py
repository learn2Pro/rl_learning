import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 创建 Breakout 游戏环境
env = gym.make('Breakout-v0', render_mode='human')

# 获取动作空间大小
w, h, dim = env.observation_space.shape
num_actions = env.action_space.n

# 定义神经网络模型
print(w, h, dim, num_actions)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(w*h*dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建 Q 网络和目标网络
q_network = DQN()
target_network = DQN()

# 定义优化器
optimizer = optim.Adam(q_network.parameters())

# 定义训练参数
num_episodes = 1000
batch_size = 32
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.99

# 进行训练
try:
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False

        while not done:
            # 展示当前状态
            env.render()

            # 将状态转换为张量
            state_tensor = torch.tensor(
                state.flatten(), dtype=torch.float).unsqueeze(0)

            # 在选择动作时使用探索率
            if random.random() <= epsilon:
                # 探索性选择随机动作
                action = env.action_space.sample()
            else:
                # 利用 Q 网络选择最优动作
                with torch.no_grad():
                    q_values = q_network(state_tensor)
                    action = q_values.argmax(dim=1).item()

            # 执行选定的动作
            next_state, reward, done, _, _ = env.step(action)

            # 将下一个状态转换为张量
            next_state_tensor = torch.tensor(
                next_state.flatten(), dtype=torch.float).unsqueeze(0)

            # 计算目标 Q 值
            with torch.no_grad():
                target_q_values = target_network(next_state_tensor)
                max_target_q_values = target_q_values.max(
                    dim=1, keepdim=True)[0]
                target_q_values = reward + discount_factor * \
                    max_target_q_values * (1 - done)

            # 计算当前状态的预测 Q 值
            q_values = q_network(state_tensor)
            action_tensor = torch.nn.functional.one_hot(
                torch.tensor(action), num_classes=num_actions).unsqueeze(0)

            # print(action, q_values.shape, action_tensor.shape)
            predicted_q_values = q_values.gather(1, action_tensor)

            # 计算损失函数
            loss = nn.MSELoss()(predicted_q_values, target_q_values)

            # 执行反向传播和优化步骤
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新状态
            state = next_state

        # 衰减探索率
        epsilon *= epsilon_decay
except KeyboardInterrupt:
    print("Ctrl+C")
finally:
    env.close()
