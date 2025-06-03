from torch.distributions import Categorical
from collections.abc import Callable, Iterable, Mapping
from typing import Any
import os
import torch.multiprocessing as mp
import time
import random
import torch
from typing import Any, Callable, Optional, Tuple
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.distributions import Beta, TransformedDistribution, AffineTransform
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import opacus
import numpy as np
from collections import deque
import random
# import os
import copy
import warnings
import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.float32)  # 设置默认数据类型为 Float
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"  # 服务器有4块显卡，选空闲的用
# visible_devices = [1, 2]  # 物理GPU编号

# 一些超参数
L = 73
N = 10  # 参与联邦学习用户数量
batch_size = 256  # 用户本地更新的batch size
episodes = 800  # 强化学习的episode数
CR_Total = 20  # 在一个episode中的联邦学习的通信轮次（客户端和服务器）
rho_min = 2.0  # 用户随机初始化的最小的总隐私预算
rho_max = 6.0  # 用户随机初始化的最大的总隐私预算
epoch_local = 1  # 一轮本地更新的epoch数
DRL_steps = 1  # 强化学习中收集一次训练数据后更新网络的次数
sigma_max = 2.0  # 最大的选取的sigma 21.6
sigma_min = 0.5  # 最小的选取的sigma 1.8
rho_used_min = 2 * (L ** 2) / ((batch_size ** 2) * (sigma_max ** 2))  # 最大sigma对应的rho
rho_used_max = 2 * (L ** 2) / ((batch_size ** 2) * (sigma_min ** 2))  # 最小sigma对应的rho


def rho2sigma(rho, bs, L):
    return np.sqrt(2 * (L ** 2) / (rho * (bs ** 2)))


def sigma2rho(sigma, bs, L):
    return 2 * (L ** 2) / ((bs ** 2) * (sigma ** 2))


# MNIST数据集的子类，把数据集（比较小）预先存入显存并做预处理
class CUDAMNIST(torchvision.datasets.MNIST):
    # 初始化函数，继承自MNIST数据集
    def __init__(self,
                 root: str,  # 数据集的根目录
                 train: bool = True,  # 是否为训练数据集
                 pre_transform: Callable[..., Any] = None,  # 数据预处理函数
                 transform: Callable[..., Any] = None,  # 数据变换函数
                 target_transform: Callable[..., Any] = None,  # 标签变换函数
                 download: bool = False,  # 是否自动下载数据集
                 device: str = 'cuda:1') -> None:  # 数据加载到哪个GPU设备上
        # 调用父类的初始化函数，并传入参数
        super().__init__(root, train, transform, target_transform, download)
        # 将数据转换为浮点数类型
        self.data = self.data.type(torch.FloatTensor)
        # 对每个样本进行预处理，预处理函数由pre_transform指定
        for i in range(len(self)):
            self.data[i] = pre_transform((self.data[i] / 255.0).numpy())  # self.data[i]/255.0 → 进行归一化
            # 将标签转换为Tensor类型，并去除额外维度
            self.targets[i] = torch.Tensor([self.targets[i]]).squeeze_().long()
        # 将数据和标签转换为适合CUDA的格式，并加载到指定的GPU设备上
        self.data = self.data.unsqueeze_(1).cuda(device=device)
        self.targets = self.targets.cuda(device=device)

    # 重写getitem函数，返回处理后的数据和标签
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


# 联邦学习神经网络模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 第一个卷积层，输入通道1，输出通道6，卷积核大小5x5
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，核大小2x2，步长2
        self.conv2 = nn.Conv2d(6, 16, 4)  # 第二个卷积层，输入通道6，输出通道16，卷积核大小4x4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 第一个全连接层，输入16*4*4，输出120
        self.fc2 = nn.Linear(120, 64)  # 第二个全连接层，输入120，输出64
        self.fc3 = nn.Linear(64, 10)  # 第三个全连接层，输入64，输出10（对应10个类别）

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        '''x = torch.flatten(x, 1)这行代码就是执行展平操作。
        参数1表示展平操作的起始维度,即除了第一个维度(batch维度)之外的其他所有维度。
        因此,如果输入x的形状是(batch_size, channels, height, width),
        那么展平操作后x的形状将会是(batch_size * channels * height * width, 1)，即一个一维向量。'''
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x)
        return x


# 定义策略网络
# class PolicyNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=64):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return torch.softmax(self.fc3(x), dim=-1)

# class PolicyNetwork(torch.nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(PolicyNetContinuous, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, 32)
#         self.fc_mu = torch.nn.Linear(32, action_dim)
#         self.fc_std = torch.nn.Linear(32, action_dim)

#     # def forward(self, x):
#     #     x = F.relu(self.fc1(x))
#     #     mu = 2.0 * torch.tanh(self.fc_mu(x))
#     #     std = F.softplus(self.fc_std(x))
#     #     return mu, std

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         mu = torch.tanh(self.fc_mu(x)) * (rho_used_max - rho_used_min)/2 + (rho_used_max + rho_used_min)/2  # [-1,1] → [val_min, val_max]
#         std = F.softplus(self.fc_std(x)) + 1e-5  # 保证σ>0
#         return mu, std

# class PolicyNetwork(nn.Module):
#     def __init__(self, dim_state, dim_action):
#         super().__init__()
#         self.fc1 = nn.Linear(dim_state, 32)
#         self.fc2 = nn.Linear(32, 16)
#         self.fc3 = nn.Linear(16, dim_action)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.sigmoid(self.fc3(x))
#         return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # self.min = min_val
        # self.max = max_val

        # 共享特征层
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        # 输出 α 和 β 参数（需保证 >1）
        self.alpha_head = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softplus()
        )

    def forward(self, x):
        features = self.feature_net(x)
        alpha = self.alpha_head(features) + 1e-3  # α ≥1
        beta = self.beta_head(features) + 1e-3  # β ≥1
        # 参数裁剪
        alpha = torch.clamp(alpha, min=1e-3, max=1e3)
        beta = torch.clamp(beta, min=1e-3, max=1e3)
        # 创建基础Beta分布
        base_dist = Beta(alpha, beta)

        # 线性变换到目标区间
        return TransformedDistribution(
            base_dist,
            [AffineTransform(loc=rho_used_min, scale=rho_used_max - rho_used_min)]
        )


# 定义价值网络
# class ValueNetwork(nn.Module):
#     def __init__(self, state_dim, hidden_dim=64):
#         super(ValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# Critic网络，dim_state为state的维度，dim_action为动作的维度
class ValueNetwork(nn.Module):
    def __init__(self, dim_state):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 32)  # 定义第一个全连接层，输入维度为状态维度加动作维度，输出维度为32
        self.fc2 = nn.Linear(32, 16)  # 定义第二个全连接层，输入维度为32，输出维度为16
        self.fc3 = nn.Linear(16, 1)  # 定义第三个全连接层，输入维度为16，输出维度为1

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 对第一层的输出应用ReLU激活函数
        x = F.relu(self.fc2(x))  # 对第二层的输出应用ReLU激活函数
        x = self.fc3(x)  # 第三层的输出不使用激活函数
        return x  # 返回最终输出


# MAPPO主类
class MAPPO:
    def __init__(self, state_dim, action_dim, n_agents, gamma=0.99, clip_epsilon=0.2, lr=1e-4,
                 device='cuda:1', explore=1.0,  # 探索的初始倾向
                 explore_decay=0.9996,  # 探索倾向的衰减率
                 ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.device = device
        self.train_step = 0  # 初始化训练步数计数器
        # self.explore = explore  # 设置初始探索率
        # self.explore_decay = explore_decay  # 设置探索率衰减系数
        self.dim_action = action_dim
        self.dim_state = state_dim
        # 初始化策略网络和价值网络
        self.policies = [PolicyNetwork(state_dim, action_dim).cuda(device=device) for _ in range(n_agents)]
        self.values = [ValueNetwork(state_dim + 3 * (n_agents - 1)).cuda(device=device) for _ in range(n_agents)]

        self.optimizers = [optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=lr)
                           for policy, value in zip(self.policies, self.values)]

        # 经验缓冲区
        self.buffer = [[] for _ in range(n_agents)]

    def store_experience(self, state, action, reward, next_state, done):
        s1 = state[0][:2]
        s2 = [s[2:] for s in state]
        global_state = s1 + [item for sublist in s2 for item in sublist]

        s1 = next_state[0][:2]
        s2 = [s[2:] for s in next_state]
        global_next_state = s1 + [item for sublist in s2 for item in sublist]
        for agent_idx in range(N):
            # 同时存储局部和全局状态
            self.buffer[agent_idx].append((state[agent_idx], global_state, action[agent_idx], reward[agent_idx],
                                           global_next_state, done[agent_idx]))

    # def compute_advantages(self, rewards, values, dones, last_value):
    #     advantages = torch.zeros_like(rewards)
    #     last_advantage = 0
    #     for t in reversed(range(len(rewards))):
    #         if dones[t]:
    #             delta = rewards[t] - values[t]
    #             last_advantage = 0
    #         else:
    #             delta = rewards[t] + self.gamma * last_value - values[t]
    #         advantages[t] = delta + self.gamma * last_advantage
    #         last_value = values[t]
    #         last_advantage = advantages[t]
    #     return advantages

    def compute_advantages(self, rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]  # 终止状态mask
            delta = rewards[t] + gamma * next_values[t] * mask - values[t]
            advantages[t] = delta + gamma * lambda_ * mask * last_advantage
            last_advantage = advantages[t]

        return advantages

    def update(self):
        # self.explore = max(self.explore * self.explore_decay, 0.01)  # 更新探索率，但不低于0.01
        for agent_idx in range(self.n_agents):
            if len(self.buffer[agent_idx]) == 0:
                continue

            # 转换数据为张量
            states, global_state, actions, rewards, next_states, dones = zip(*self.buffer[agent_idx])
            states = torch.FloatTensor(np.array(states)).cuda(device=self.device)
            global_state = torch.FloatTensor(np.array(global_state)).cuda(device=self.device)
            actions = torch.FloatTensor(np.array(actions)).cuda(device=self.device)
            rewards = torch.FloatTensor(np.array(rewards)).cuda(device=self.device)
            next_states = torch.FloatTensor(np.array(next_states)).cuda(device=self.device)
            dones = torch.FloatTensor(np.array(dones)).cuda(device=self.device)

            actions = torch.clamp(
                actions,
                min=rho_used_min + 1e-5,
                max=rho_used_max - 1e-5
            )
            # 计算优势估计
            # with torch.no_grad():
            #     values = self.values[agent_idx](global_state).squeeze()
            #     last_value = self.values[agent_idx](next_states[-1]).item()

            # advantages = self.compute_advantages(rewards, values,
            #                                      dones, last_value)
            with torch.no_grad():
                values = self.values[agent_idx](global_state).squeeze()
                next_values = self.values[agent_idx](next_states).squeeze()
                next_values = next_values * (1 - dones)  # 终止状态后价值置零

            # 计算优势
            advantages = self.compute_advantages(rewards, values, next_values, dones)

            # 计算旧策略概率
            with torch.no_grad():  # 确保整个块不追踪梯度
                action_dist = self.policies[agent_idx](states)
                # action = action_dist.rsample()            # 重参数化采样（可导）
                old_log_probs = action_dist.log_prob(actions)  # 对数概率

            # PPO优化步骤
            for _ in range(3):  # 通常进行3-10次epoch
                # self.explore = max(self.explore * self.explore_decay, 0.01)  # 更新探索率，但不低于0.01
                # 计算新策略概率
                new_dist = self.policies[agent_idx](states)
                new_log_probs = new_dist.log_prob(actions)

                # 计算PPO损失
                ratio = (new_log_probs - old_log_probs).exp()
                # new_probs = self.policies[agent_idx](states).gather(1, actions.unsqueeze(1))
                # ratio = (new_probs / old_probs).squeeze()

                # 计算裁剪损失
                surr1 = ratio * advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                value_loss = nn.MSELoss()(self.values[agent_idx](global_state).squeeze(),
                                          rewards + self.gamma * (1 - dones) * self.values[agent_idx](
                                              next_states).detach().squeeze())

                # 熵正则化
                # entropy = Categorical(self.policies[agent_idx](states)).entropy().mean()
                # 熵奖励
                # entropy = new_dist.entropy().mean()
                base_entropy = new_dist.base_dist.entropy()  # 基础Beta分布的熵
                scale = rho_used_max - rho_used_min
                entropy = (base_entropy + torch.log(torch.tensor(scale, device=self.device))).mean()

                # 总损失
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # 优化步骤
                self.optimizers[agent_idx].zero_grad()
                loss.backward()
                self.optimizers[agent_idx].step()

            # 清空缓冲区
            self.buffer[agent_idx] = []
        self.train_step += 1

    # 无噪声地输出一个action，用于测试
    # def take_action(self, state, idx):
    #     with torch.no_grad():  # 禁用梯度计算，减少内存使用并加速计算
    #         action = self.policies[idx](torch.Tensor(state).cuda(
    #             device=self.device)).cpu().numpy()  # 将状态转换为tensor，送入GPU，通过actor网络计算动作，然后转回CPU并转为numpy数组
    #         return action  # 返回计算得到的动作

    def take_action(self, state, idx):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_dist = self.policies[idx](state)
            base_mean = action_dist.base_dist.mean
            action_mean = rho_used_min + (rho_used_max-rho_used_min) * base_mean
            # action_dist = TransformedDistribution(
            #     base_dist,
            #     transforms=[
            #         AffineTransform(
            #             loc=rho_used_min,
            #             scale=rho_used_max-rho_used_min
            #         )
            #     ]
            # )
            return action_mean.cpu().numpy()

            # state = torch.tensor([state], dtype=torch.float).to(self.device)
            # mu, sigma = self.actor(state)
            # low = torch.tensor([rho_used_min])
            # high = torch.tensor([rho_used_max])
            # action_dist = TruncatedNormal(mu, sigma, low, high)
            # action = action_dist.sample()
            # return [action.item()]

    # 有噪声地输出一个action，用于训练
    def take_action_with_noise(self, state, idx):
        with torch.no_grad():  # 禁用梯度计算
            state_tensor = torch.FloatTensor(state).to(self.device)

            # 生成动作分布
            action_dist = self.policies[idx](state_tensor)

            # 采样动作（训练模式：探索）
            action = action_dist.rsample()  # 重参数化采样（可导）
            # log_prob = action_dist.log_prob(action)  # 对数概率
            action = torch.clamp(
                action,
                min=rho_used_min + 1e-5,
                max=rho_used_max - 1e-5
            )
            # 转换为环境可接受的格式
            return action.cpu().numpy()

            # state = torch.tensor([state], dtype=torch.float).to(self.device)
            # mu, sigma = self.actor(state)
            # low = torch.tensor([rho_used_min])
            # high = torch.tensor([rho_used_max])
            # action_dist = TruncatedNormal(mu, sigma, low, high)
            # action = action_dist.sample()
            # explore = self.explore  # 获取当前的探索参数
            # action_noise = action + np.random.normal(loc=0.0, scale=explore, size=self.dim_action)  # 给动作添加高斯噪声
            # for i in range(len(action_noise)):  # 对每个动作维度进行处理
            #     local_explore = explore  # 初始化局部探索参数
            #     while action_noise[i] < rho_used_min or action_noise[i] > rho_used_max:  # 如果动作超出范围
            #         local_explore /= 2  # 将局部探索参数减半
            #         if local_explore <= 0.0001:  # 如果局部探索参数太小
            #             action_noise[i] = action[i]  # 直接使用原始动作
            #             break
            #         action_noise[i] = np.random.normal(loc=action_noise[i], scale=local_explore, size=1)  # 重新生成噪声
            #     # print(action_noise[0])
            # return action, action_noise  # 返回原始动作和带噪声的动作

    # 保存模型
    def save(self, episode):
        save_dir = './MAPPO_Nets'
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        for i in range(self.n_agents):
            torch.save(self.policies[i].state_dict(), os.path.join(save_dir, 'actor_' + str(i)
                                                                   + '_' + str(episode) + '.pth'))

            torch.save(self.values[i].state_dict(),
                       os.path.join(save_dir, 'critic_' + str(i) + '_' + str(episode) + '.pth'))  # 保存critic网络的参数

    # 加载模型
    def load(self, episode):
        for i in range(self.n_agents):
            self.policies[i].load_state_dict(
                torch.load('./MAPPO_Nets/actor_' + str(i) + '_' + str(episode) + '.pth', map_location=self.device))

            self.values[i].load_state_dict(
                torch.load('./MAPPO_Nets/critic_' + str(i) + '_' + str(episode) + '.pth', map_location=self.device))


# 联邦学习用户的训练进程(神经网络训练)
class Training(mp.Process):
    def __init__(self,
                 C=4,  # clipping bound
                 bs=batch_size,  # 本地更新的batch size
                 lr=0.1,  # 学习率
                 el=epoch_local,  # 一轮本地更新的epoch数
                 ins=mp.Queue(),  # 输入队列，用于进程间通信
                 outs=mp.Queue(),  # 输出队列，用于进程间通信
                 device='cuda:1'  # 选择的gpu设备
                 ) -> None:
        mp.Process.__init__(self)

        # 用户参数
        self.device = device
        self.C = C
        self.bs = bs
        self.el = el
        self.lr = lr
        self.trainloader = None
        self.trset = None

        # Processes communication
        self.ins = ins
        self.outs = outs

    # 进程运行
    def run(self):
        # 初始化用户数据集与网络
        self.trset = self.ins.get()  # 从输入队列获取训练集
        self.trainloader = torch.utils.data.DataLoader(self.trset, batch_size=self.bs,
                                                       shuffle=True, num_workers=0, pin_memory=False)  # 创建数据加载器
        self.net = Net()  # 初始化神经网络模型
        pe = opacus.PrivacyEngine()  # 创建隐私引擎实例
        self.criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)  # 定义优化器为SGD
        self.net, self.optimizer, _ = pe.make_private(
            module=self.net,
            data_loader=self.trainloader,
            optimizer=self.optimizer,
            noise_multiplier=1.0,
            max_grad_norm=self.C
        )  # 使用隐私引擎将模型、优化器和数据加载器转换为隐私版本
        self.net.cuda(device=self.device)  # 将模型移动到指定的GPU设备上
        print('Dataset loaded successfully of', self.device)  # 打印数据集加载成功的消息
        self.outs.put(1)  # 向输出队列发送信号，表示初始化完成

        # 根据输入的mode选择要进行的工作：结束进程，训练，测试
        while True:
            mode, net, sigma = self.ins.get()  # 从输入队列获取模式、网络和sigma值
            if mode == 0:  # Terminate mode
                break  # 如果是终止模式，退出循环

            elif mode == 1:  # training mode
                self.net.load_state_dict(net.state_dict())  # 加载传入的网络参数
                self.optimizer.noise_multiplier = sigma  # 设置优化器的噪声乘数
                for epoch in range(self.el):  # 对每个本地训练轮次
                    for i, data in enumerate(self.trainloader):  # 遍历训练数据
                        self.optimizer.zero_grad()  # 清零梯度
                        inputs, labels = data  # 获取输入和标签
                        outputs = self.net(inputs)  # 前向传播
                        loss = self.criterion(outputs, labels)  # 计算损失
                        loss.backward()  # 反向传播
                        self.optimizer.step()  # 更新参数
                self.outs.put(self.net)  # 将训练后的网络放入输出队列
            else:  # Testing mode
                self.net.load_state_dict(net.state_dict())  # 加载传入的网络参数
                with torch.no_grad():  # 不计算梯度
                    correct = 0  # 正确预测的样本数
                    total = 0  # 总样本数
                    avg_loss = 0  # 平均损失
                    count = 0  # 批次计数
                    total_loss = 0
                    for i, data in enumerate(self.trainloader):  # 遍历训练数据
                        inputs, labels = data  # 获取输入和标签
                        outputs = self.net(inputs)  # 前向传播
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            print("模型输出包含 NaN 或 inf！")
                            raise ValueError("模型输出异常")
                        loss = self.criterion(outputs, labels)  # 计算损失
                        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
                        total += labels.size(0)  # 更新总样本数
                        correct += (predicted == labels).sum().item()  # 更新正确预测数
                        # avg_loss = avg_loss * (count / (count + 1)) + loss.item() / (count + 1)  # 更新平均损失
                        total_loss += loss.item()
                        count += 1  # 更新批次计数
                    avg_loss = total_loss / len(self.trainloader)
                    acc = correct / total  # 计算准确率
                    self.outs.put([acc, avg_loss])  # 将准确率和平均损失放入输出队列


# 联邦学习Client
class Client():
    def __init__(self,
                 id=-1,  # client id
                 C=4,  # clipping bound
                 rho_total=20.0,  # client总的隐私预算
                 bs=batch_size,  # 本地更新batch size
                 trset=None,  # 用户的数据集
                 lr=0.1,  # 本地更新学习率
                 el=epoch_local,  # 一轮本地更新的epoch数
                 hl=5,  # 历史的长度（未使用）
                 agent=None,  # 用户的DRL agent
                 device='cuda:1'  # 选择的gpu设备
                 ) -> None:
        # 用户的信息
        self.id = id
        self.rho = rho_total
        self.rho_total = rho_total
        self.C = C
        self.bs = bs
        self.trset = trset
        self.lr = lr
        self.el = el
        self.hl = hl
        self.sigma = sigma_max  # 选择的sigma
        self.rho_used = 2 * self.el / (self.sigma ** 2)  # sigma对应的使用的隐私预算
        self.local_acc = 0.1  # 记录用本地数据集测试的精度

        # DRL variables
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = None
        self.real_action = None

        self.device = device
        self.agent = agent

        # 训练进程相关设置
        self.ins = mp.Queue(1)  # 创建一个输入队列，用于向训练进程发送数据
        self.outs = mp.Queue(1)  # 创建一个输出队列，用于从训练进程接收结果
        self.process = Training(C=self.C, bs=self.bs, lr=self.lr, el=self.el,
                                ins=self.ins, outs=self.outs, device=device)  # 创建训练进程
        self.process.start()  # 启动训练进程
        self.ins.put(self.trset)  # 将训练集放入输入队列
        # self.L = self.outs.get()  # 等待数据加载完成的信号
        self.outs.get()

    # 除了数据集外重置用户的信息
    def reset(self, rho_total):
        self.rho = rho_total  # 重置当前隐私预算
        self.rho_total = rho_total  # 重置总隐私预算
        self.action = None  # 重置动作
        self.sigma = sigma_max  # 重置噪声参数为最大值
        self.local_acc = 0.1  # 重置本地准确率

    # 训练方法
    def train(self, net, sigma, rho_used):
        self.sigma = sigma  # 设置当前噪声参数
        # print(sigma)
        self.rho_used = rho_used
        self.rho -= self.rho_used  # 更新剩余隐私预算
        self.ins.put([1, net, sigma])  # 将训练指令、模型和噪声参数放入输入队列

    # 测试方法
    def test(self, net):
        self.ins.put([2, net, 0])  # 将测试指令和模型放入输入队列

    # 终止进程方法
    def terminate_process(self):
        self.ins.put([0, 0, 0])  # 发送终止信号到输入队列


# 服务器类
class Server():
    def __init__(self,
                 net,  # 初始化全局神经网络
                 device='cuda:1'  # 选择的gpu设备
                 ) -> None:
        self.device = device  # 设置设备
        self.net = Net()  # 创建一个新的神经网络实例
        self.net.load_state_dict(net.state_dict())  # 加载传入的网络参数
        dummyOptimizer = optim.SGD(self.net.parameters(), lr=0)  # 创建一个学习率为0的SGD优化器（仅用于隐私引擎）
        dummySet = [[0, 0]]  # 创建一个虚拟数据集
        dummyLoader = torch.utils.data.DataLoader(dummySet)  # 创建一个虚拟数据加载器
        privacyEngine = opacus.PrivacyEngine()  # 初始化隐私引擎
        self.net, _, _ = privacyEngine.make_private(  # 使用隐私引擎将网络转换为私有版本
            module=self.net,
            optimizer=dummyOptimizer,
            data_loader=dummyLoader,
            noise_multiplier=4,
            max_grad_norm=4
        )
        self.dummy_net = copy.deepcopy(self.net)  # 创建网络的深拷贝作为临时网络
        self.dummy_net.cuda(device=self.device)  # 将临时网络移到指定设备
        self.net.cuda(device=self.device)  # 将主网络移到指定设备

    def reset(self, net):
        self.net = Net()  # 创建一个新的神经网络实例
        self.net.load_state_dict(net.state_dict())  # 加载传入的网络参数
        dummyOptimizer = optim.SGD(self.net.parameters(), lr=0)  # 创建一个学习率为0的SGD优化器（仅用于隐私引擎）
        dummySet = [[0, 0]]  # 创建一个虚拟数据集
        dummyLoader = torch.utils.data.DataLoader(dummySet)  # 创建一个虚拟数据加载器
        privacyEngine = opacus.PrivacyEngine()  # 初始化隐私引擎
        self.net, _, _ = privacyEngine.make_private(  # 使用隐私引擎将网络转换为私有版本
            module=self.net,
            optimizer=dummyOptimizer,
            data_loader=dummyLoader,
            noise_multiplier=4,
            max_grad_norm=4
        )
        self.net.cuda(device=self.device)  # 将网络移到指定设备

    # 聚合全局模型
    def aggregate(self, net_list):
        count = 0  # 初始化计数器
        for net in net_list:  # 遍历所有客户端的网络
            self.dummy_net.load_state_dict(net.state_dict())  # 将当前客户端的网络参数加载到临时网络
            for name, p in self.net.named_parameters():  # 遍历主网络的所有参数
                # 使用加权平均更新主网络的参数
                p.data = p.data * (count / (count + 1)) + self.dummy_net.state_dict()[name].data / (count + 1)
            count += 1  # 增加计数器


if __name__ == '__main__':
    torch.cuda.empty_cache()
    t0 = time.time()  # 记录程序开始时间
    agent = MAPPO(n_agents=N,  # 初始化Agent对象，设置智能体数量为N
                  state_dim=5,  # 设置状态空间维度为5
                  action_dim=1)  # 设置动作空间维度为1
    # agent.load(108)
    mp.set_start_method('spawn', force=True)  # 设置多进程启动方法为'spawn'，确保在Windows和Linux上的兼容性

    # 初始化 server
    net = Net()  # 创建一个新的神经网络实例
    server = Server(net=net)  # 使用创建的网络初始化服务器

    # 分割 Datasets
    transform = transforms.Compose(  # 定义数据预处理流程
        [transforms.ToTensor(),  # 将图像转换为PyTorch张量
         transforms.Normalize((0.5), (0.5))])  # 标准化图像，使像素值在[-1, 1]范围内
    # batch_size = 200  # 设置批处理大小
    trainsets = []  # 初始化训练集列表
    for i in range(torch.cuda.device_count()):  # 遍历所有可用的CUDA设备
        # device_index = visible_devices[i % len(visible_devices)]
        trainset = CUDAMNIST(root='./data', train=True,  # 为每个CUDA设备创建一个MNIST训练集
                             download=True, pre_transform=transform, device='cuda:' + str(i))
        trainsets.append(trainset)  # 将创建的训练集添加到列表中

    indices = list(range(len(trainsets[0])))  # 创建数据集索引列表
    random.shuffle(indices)  # 随机打乱索引顺序
    dataset_num = int(len(trainsets[0]) / N)  # 计算每个客户端分配的数据量
    trsets = []  # 初始化客户端训练集列表
    for i in range(N):  # 遍历所有客户端
        # device_index = visible_devices[i % len(visible_devices)]
        device_index = i % torch.cuda.device_count()  # 计算当前客户端使用的CUDA设备索引
        device = 'cuda:' + str(device_index)  # 构造CUDA设备字符串
        trset = torch.utils.data.Subset(trainsets[device_index],
                                        indices[i * dataset_num: (i + 1) * dataset_num])  # 创建子数据集
        trsets.append([trset, device])  # 将子数据集和对应的设备添加到列表中

    # 初始化 clients
    # 初始化客户端列表，每个客户端都是Client类的实例
    clients = [Client(id=i,  # 客户端ID
                      rho_total=np.random.uniform(low=rho_min, high=rho_max),  # 随机生成总隐私预算
                      trset=trsets[i][0],  # 分配给该客户端的训练数据集
                      agent=agent,  # 共享的Agent实例
                      device=trsets[i][1]) for i in range(N)]  # 分配的GPU设备

    # 开始深度强化学习（DRL）训练
    episode = 1  # 初始化训练轮次
    final_accs = []  # 存储每轮训练结束时的最终准确率
    mean_final_accs = []  # 存储最近几轮的平均最终准确率
    returns = []  # 存储每轮的回报（准确率提升）
    mean_returns = []  # 存储最近几轮的平均回报
    high_acc = 0.0  # 记录最高准确率，用于保存最佳模型
    while episode < episodes:  # 主训练循环，直到达到预设的轮次
        t1 = time.time()  # 记录本轮开始时间
        print('Trained steps', agent.train_step)  # 打印已训练的步数
        # print('Explore ', agent.explore)  # 打印当前探索率
        print('Episode', episode)  # 打印当前轮次

        # 重置客户端和服务器状态
        net_init = Net()  # 创建新的初始网络
        net = Net()  # 创建新的当前网络
        net.load_state_dict(net_init.state_dict())  # 将初始网络的参数复制到当前网络
        server.reset(net=net)  # 重置服务器，使用新的网络
        rhos_init = []  # 初始化隐私预算列表
        for i, client in enumerate(clients):
            rho = np.random.uniform(low=rho_min, high=rho_max)  # 为每个客户端随机生成新的隐私预算
            rhos_init.append(rho)  # 将新的隐私预算添加到列表
            client.reset(rho_total=rho)  # 重置客户端，使用新的隐私预算
            # print(client.rho_used)
        sigmas = [[] for _ in range(N)]  # 初始化每个客户端的噪声列表
        rho_useds = [[] for _ in range(N)]  # 初始化每个客户端的已使用隐私预算列表
        losses = [[] for _ in range(N)]  # 每个客户每轮的损失

        # 使用初始全局模型进行测试并记录准确率
        global_accs = []  # 初始化全局准确率列表
        global_losses = []
        for client in clients:
            client.test(server.net)  # 每个客户端使用服务器的网络进行测试
        last_global_loss = 0  # 初始化上一轮的全局损失
        last_global_acc = 0  # 初始化上一轮的全局准确率
        for i, client in enumerate(clients):
            acc, avg_loss = client.outs.get()  # 获取每个客户端的测试结果
            losses[i].append(avg_loss)
            last_global_loss += avg_loss  # 累加损失
            last_global_acc += acc  # 累加准确率
        last_global_loss /= N  # 计算平均损失
        last_global_acc /= N  # 计算平均准确率
        global_accs.append(last_global_acc)  # 将初始全局准确率添加到列表
        global_losses.append(last_global_loss)

        isDone = 0.0  # 初始化完成标志为0
        for CR in range(CR_Total):  # 遍历每一轮通信
            if CR == CR_Total - 1:  # 如果是最后一轮
                isDone = 1.0  # 设置完成标志为1
            sigma_avg = 0  # 初始化平均sigma值
            avgloss = []
            flag = -1
            for i, client in enumerate(clients):  # 遍历每个客户端

                state = [CR, CR_Total - CR, client.rho_total - client.rho, client.rho, losses[i][CR]]
                client.state = state  # 设置客户端状态
                action_noise = agent.take_action_with_noise(state=state, idx=i)  # 获取带噪声的动作
                rho_used = action_noise[0]  # 将动作映射到sigma值
                sigma = rho2sigma(rho_used, client.bs, L)

                sigmas[i].append(sigma)  # 将sigma值添加到列表
                rho_useds[i].append(rho_used)  # 将使用的隐私预算添加到列表
                client.sigma = sigmas[i][CR]  # 设置客户端的sigma值

                client.action = (rho_used - rho_used_min) / (rho_used_max - rho_used_min)

                client.train(net=server.net, sigma=client.sigma, rho_used=rho_used)  # 使用当前sigma值训练客户端模型

            net_list = []  # 初始化网络列表
            for client in clients:  # 遍历每个客户端
                cnet = client.outs.get()  # 获取客户端训练后的模型
                net_list.append(cnet)  # 将模型添加到列表中
            server.aggregate(net_list=net_list)  # 聚合所有客户端的模型
            for client in clients:  # 遍历每个客户端
                client.test(server.net)  # 使用聚合后的模型进行测试

            global_loss = 0  # 初始化全局损失
            global_acc = 0  # 初始化全局准确率
            for i, client in enumerate(clients):  # 遍历每个客户端
                acc, avg_loss = client.outs.get()  # 获取客户端的测试结果
                # assert not np.isnan(avg_loss).any(), "avgLoss 输出 NaN！"
                losses[i].append(avg_loss)
                global_loss += avg_loss  # 累加损失
                global_acc += acc  # 累加准确率
                client.local_acc = acc  # 设置客户端的本地准确率
            global_loss /= N  # 计算平均全局损失
            global_acc /= N  # 计算平均全局准确率

            states = [client.state for client in clients]  # 收集所有客户端的状态
            actions = [client.action for client in clients]  # 收集所有客户端的动作
            # rewards = [global_acc - last_global_acc]  # 计算奖励（全局准确率的提升）
            rewards = []
            for i, client in enumerate(clients):
                exceed = 0
                # if CR != 0:
                #     exceed = rho_useds[i][CR]-rho_useds[i][CR-1]
                # rewards.append(avgloss[i] - losses[i][CR + 1] + exceed)
                if CR == CR_Total - 1:
                    if client.rho < 0:
                        rewards.append(2 * (losses[i][CR] - losses[i][CR + 1]) - 2 * abs(client.rho))
                    else:
                        rewards.append(2 * (losses[i][CR] - losses[i][CR + 1]) - abs(client.rho))
                else:
                    # temp = client.rho - (CR_Total - CR - 2) * rho_useds[i][CR]
                    # if temp < rho_used_min:
                    #   exceed = 3 * abs(temp - rho_used_min)
                    rewards.append(2 * (losses[i][CR] - losses[i][CR + 1]))

            global_accs.append(global_acc)  # 记录全局准确率
            last_global_acc = global_acc  # 更新上一轮的全局准确率
            global_losses.append(global_loss)
            last_global_loss = global_loss

            next_states = [[CR + 1, CR_Total - (CR + 1), client.rho_total - client.rho, client.rho,
                            losses[i][CR + 1]]
                           for i, client in enumerate(clients)]

            # next_states = [[CR + 1, CR_Total - (CR + 1), client.rho_total - client.rho, client.rho,
            #                 sum(losses[i])/len(losses[i])]
            #    for i, client in enumerate(clients)]
            done = [isDone for _ in range(N)]  # 设置完成标志

            agent.store_experience(states, actions, rewards, next_states, done)
            # agent.train(state=states, action=actions, reward=rewards, next_state=next_states, done=done)  # 训练DRL代理

            # if flag != -1:
            #     break
        agent.update()
        print("update")
        rho_used_sums = [sum(rho_useds[i]) for i in range(N)]

        # 测试，每训练1个episode进行一次测试
        if episode % 1 == 0:  # 每个episode都执行以下操作
            net_init = Net()  # 初始化一个新的神经网络
            net = Net()  # 再初始化一个新的神经网络
            net.load_state_dict(net_init.state_dict())  # 将初始化的网络参数复制到第二个网络
            server.reset(net=net)  # 重置服务器，使用新的网络
            rhos_init = []  # 初始化一个空列表，用于存储每个客户端的初始隐私预算

            for client in clients:  # 为每个客户端生成一个随机的初始隐私预算
                rho = np.random.uniform(low=rho_min, high=rho_max)  # 在给定范围内随机生成隐私预算
                rhos_init.append(rho)  # 将生成的隐私预算添加到列表中

            sigmas = [[] for _ in range(N)]  # 初始化一个嵌套列表，用于存储每个客户端在每轮通信中的sigma值
            rho_useds = [[] for _ in range(N)]  # 初始化一个嵌套列表，用于存储每个客户端在每轮通信中使用的rho值

            for i, client in enumerate(clients):  # 再次遍历每个客户端
                client.reset(rho_total=rhos_init[i])  # 使用初始的隐私预算重置客户端

            global_accs_DRL = []  # 初始化一个列表，用于存储DRL方法的全局准确率
            losses_DRL = [[] for _ in range(N)]

            for client in clients:  # 遍历每个客户端
                client.test(server.net)  # 使用服务器的网络测试客户端
            last_global_loss = 0  # 初始化上一轮的全局损失
            last_global_acc = 0  # 初始化上一轮的全局准确率
            for i, client in enumerate(clients):  # 遍历每个客户端
                acc, avg_loss = client.outs.get()  # 获取客户端的测试结果
                losses_DRL[i].append(avg_loss)
                last_global_loss += avg_loss  # 累加损失
                last_global_acc += acc  # 累加准确率
            last_global_loss /= N  # 计算平均全局损失
            last_global_acc /= N  # 计算平均全局准确率
            global_accs_DRL.append(last_global_acc)  # 记录初始的全局准确率

            for CR in range(CR_Total):  # 遍历每一轮通信
                for i, client in enumerate(clients):  # 遍历每个客户端
                    state = [CR, CR_Total - CR, client.rho_total - client.rho, client.rho, losses_DRL[i][CR]]
                    action = agent.take_action(state=state, idx=i)  # 根据当前状态，让agent选择一个动作
                    rho_used = action[0]  # 将动作转换为实际的sigma值
                    sigma = rho2sigma(rho_used, client.bs, L)
                    sigmas[i].append(sigma)  # 记录这个sigma值
                    rho_useds[i].append(rho_used)  # 记录这个rho值

                    client.sigma = sigma  # 设置客户端的sigma值
                    client.train(net=server.net, sigma=client.sigma, rho_used=rho_used)  # 使用当前sigma值训练客户端
                net_list = []  # 初始化一个列表，用于存储所有客户端的网络
                for client in clients:  # 遍历每个客户端
                    cnet = client.outs.get()  # 获取客户端训练后的网络
                    net_list.append(cnet)  # 将网络添加到列表中
                server.aggregate(net_list=net_list)  # 聚合所有客户端的网络
                for client in clients:  # 遍历每个客户端
                    client.test(server.net)  # 使用聚合后的网络测试客户端

                global_loss = 0  # 初始化全局损失
                global_acc = 0  # 初始化全局准确率
                for i, client in enumerate(clients):  # 遍历每个客户端
                    acc, avg_loss = client.outs.get()  # 获取客户端的测试结果
                    losses_DRL[i].append(avg_loss)
                    global_loss += avg_loss  # 累加损失
                    global_acc += acc  # 累加准确率
                    client.local_acc = acc  # 更新客户端的本地准确率
                global_loss /= N  # 计算平均全局损失
                global_acc /= N  # 计算平均全局准确率

                global_accs_DRL.append(global_acc)  # 记录当前轮次的全局准确率
                last_global_acc = global_acc  # 更新上一轮的全局准确率

            print('Original sigmas:',
                  [np.mean([sigmas[i][CR] for i in range(N)]) for CR in range(CR_Total)])  # 打印每轮通信中所有客户端的平均sigma值
            rho_used_sums = [sum(rho_useds[i]) for i in range(N)]  # 计算每个客户端使用的总rho值

            returns_DRL = [global_accs_DRL[-1] - global_accs_DRL[0]]  # 计算DRL方法的回报（最终准确率与初始准确率的差）
            final_accs.append(global_accs_DRL[-1])  # 记录最终的全局准确率
            returns.append(returns_DRL)  # 记录回报

            low_index = max(len(final_accs) - 10, 0)  # 计算最近10个准确率的起始索引
            mean_acc = np.mean(final_accs[low_index:])  # 计算最近10个准确率的平均值
            mean_final_accs.append(mean_acc)  # 记录平均准确率
            # 如果平均准确率超过历史最高，并且已经训练了超过10个episode，则保存模型
            if mean_acc > high_acc and episode > 10 or episode % 100 == 0:
                high_acc = mean_acc  # 更新最高准确率
                agent.save(episode=episode)  # 保存当前的agent模型
                print("save " + str(episode))

            low_index = max(len(returns) - 10, 0)  # 计算最近10个回报的起始索引
            mean_return = np.mean(returns[low_index:])  # 计算最近10个回报的平均值
            mean_returns.append(mean_return)  # 记录平均回报
        print('Time spent for this episode', time.time() - t1)  # 打印本episode的耗时

        torch.cuda.empty_cache()
        episode += 1  # 增加episode计数

    # Terminate all clients
    # 遍历所有客户端，发送终止信号
    for client in clients:
        client.terminate_process()  # 调用每个客户端的terminate_process方法，发送终止信号

    # 再次遍历所有客户端，等待所有客户端进程完全结束
    for client in clients:
        client.process.join()  # 等待每个客户端的进程完全结束，确保所有资源被正确释放

    # 打印整个程序的运行时间
    print('Time spend', time.time() - t0)  # 计算并打印从程序开始到现在的总运行时间
