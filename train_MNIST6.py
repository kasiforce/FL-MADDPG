from collections.abc import Callable, Iterable, Mapping
from typing import Any
import numpy as np
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"  # 服务器有4块显卡，选空闲的用

# 一些超参数
L = 103
N = 10  # 参与联邦学习用户数量
batch_size = 256  # 用户本地更新的batch size
episodes = 1000  # 强化学习的episode数
CR_Total = 20  # 在一个episode中的联邦学习的通信轮次（客户端和服务器）
rho_min = 2.0  # 用户随机初始化的最小的总隐私预算
rho_max = 6.0  # 用户随机初始化的最大的总隐私预算
epoch_local = 1  # 一轮本地更新的epoch数
DRL_steps = 1  # 强化学习中收集一次训练数据后更新网络的次数
sigma_max = 5.0  # 最大的选取的sigma 2.5
sigma_min = 0.5  # 最小的选取的sigma 1
rho_used_min = 2 * (L ** 2) / ((batch_size ** 2) * (sigma_max ** 2))  # 最大sigma对应的rho
rho_used_max = 2 * (L ** 2) / ((batch_size ** 2) * (sigma_min ** 2))  # 最小sigma对应的rho


def rho2sigma(rho, bs, L):
    return np.sqrt(2 * (L ** 2) / (rho * (bs ** 2)))


def sigma2rho(sigma, bs, L):
    return 2 * (L ** 2) / ((bs ** 2) * (sigma ** 2))


def compute_lipschitz_constant(model, inputs, targets, num_samples=100):
    # 确保模型处于训练模式
    model.train()
    L = 600.0
    # 获取模型的当前参数
    theta_n = model.state_dict()

    for _ in range(num_samples):
        # 创建两组新的参数字典，每个参数都需要梯度
        theta1 = {name: torch.randn_like(param, requires_grad=True) for name, param in theta_n.items()}
        theta2 = {name: torch.randn_like(param, requires_grad=True) for name, param in theta_n.items()}

        # 创建两个临时模型
        temp_net1 = copy.deepcopy(model)
        temp_net2 = copy.deepcopy(model)

        # 将新参数加载到临时模型中
        temp_net1.load_state_dict(theta1)
        temp_net2.load_state_dict(theta2)

        criterion = nn.CrossEntropyLoss()

        # 计算第一组参数的梯度
        temp_net1.zero_grad()
        outputs1 = temp_net1(inputs)
        loss1 = criterion(outputs1, targets)
        loss1.backward()

        # 计算第二组参数的梯度
        temp_net2.zero_grad()
        outputs2 = temp_net2(inputs)
        loss2 = criterion(outputs2, targets)
        loss2.backward()

        # 计算梯度差的二范数和参数差的二范数

        grad_diff_norm = torch.norm(
            torch.cat([param1.grad.view(-1) for param1 in temp_net1.parameters()]) - torch.cat(
                [param2.grad.data.view(-1) for param2 in temp_net2.parameters()]))
        param_diff_norm = torch.norm(
            torch.cat([p.view(-1) for p in theta1.values()]) - torch.cat([p.view(-1) for p in theta2.values()]))

        # 计算利普希兹常数L
        if param_diff_norm != 0:
            temp = grad_diff_norm / param_diff_norm
            if temp != 0:
                L = min(temp, L)

    return L.detach().numpy()


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
                 device: str = 'cuda:0') -> None:  # 数据加载到哪个GPU设备上
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


# Actor网络，dim_state为state的维度，dim_action为动作的维度
class Actor(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, dim_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


# Critic网络，dim_state为state的维度，dim_action为动作的维度
class Critic(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state + dim_action, 32)  # 定义第一个全连接层，输入维度为状态维度加动作维度，输出维度为32
        self.fc2 = nn.Linear(32, 16)  # 定义第二个全连接层，输入维度为32，输出维度为16
        self.fc3 = nn.Linear(16, 1)  # 定义第三个全连接层，输入维度为16，输出维度为1

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 对第一层的输出应用ReLU激活函数
        x = F.relu(self.fc2(x))  # 对第二层的输出应用ReLU激活函数
        x = self.fc3(x)  # 第三层的输出不使用激活函数
        return x  # 返回最终输出


# 强化学习DDPG模型
class Agent():
    def __init__(self,
                 n_agents,  # 联邦学习用户个数
                 dim_state,  # state维度
                 dim_action,  # action维度
                 lr_actor=0.00002,  # actor的学习率
                 lr_critic=0.0002,  # critic的学习率
                 lr_actor_target=0.005,  # actor目标网络的更新率
                 lr_critic_target=0.005,  # critic目标网络的更新率
                 buffer_size=4000,  # DDPG中buffer的最大大小，trajectory数量如果超过这个数会删除buffer中旧的trajectory → 经验池大小
                 minimal_size=512,  # buffer中的trajectory数量达到该值后开始更新网络
                 gamma=0.998,  # discount factor折扣系数
                 explore=1.0,  # 探索的初始倾向
                 explore_decay=0.9996,  # 探索倾向的衰减率
                 explore_decay_interval=1000,  # 探索倾向的衰减间隔（未使用）
                 bs=256,  # 进行一次DDPG更新的batch size
                 train_interval=1,  # DDPG更新间隔，影响更新频率
                 device='cuda:0'  # 选择的gpu设备
                 ) -> None:

        self.device = device
        self.n_agents = n_agents
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.actors = []
        self.actors_target = []
        self.critics = []
        self.critics_target = []
        for i in range(n_agents):
            # 初始化actor网络和目标网络
            actor = Actor(dim_state=self.dim_state, dim_action=self.dim_action).cuda(device=device)  # 创建actor网络并移至GPU
            actor_target = Actor(dim_state=self.dim_state, dim_action=self.dim_action).cuda(
                device=device)  # 创建actor目标网络并移至GPU
            actor_target.load_state_dict(actor.state_dict())  # 将actor网络的参数复制到目标网络
            self.actors.append(actor)
            self.actors_target.append(actor_target)

            critic = Critic(dim_state=self.dim_state + 3 * (n_agents - 1), dim_action=self.dim_action * n_agents).cuda(
                device=device)
            critic_target = Critic(dim_state=self.dim_state + 3 * (n_agents - 1),
                                   dim_action=self.dim_action * n_agents).cuda(
                device=device)
            critic_target.load_state_dict(critic.state_dict())
            self.critics.append(critic)
            self.critics_target.append(critic_target)

        # 设置网络优化参数
        self.lr_actor_target = lr_actor_target  # 设置actor目标网络的学习率
        self.lr_critic_target = lr_critic_target  # 设置critic目标网络的学习率
        self.bs = bs  # 设置批量大小
        self.lr_actor_target = lr_actor_target  # 再次设置actor目标网络的学习率（可能是冗余的）
        self.criterion_critic = nn.MSELoss()  # 设置critic的损失函数为均方误差
        self.optimizer_actor = [optim.Adam(self.actors[i].parameters(), lr=lr_actor) for i in
                                range(self.n_agents)]  # 创建actor的Adam优化器
        self.optimizer_critic = [optim.Adam(self.critics[i].parameters(), lr=lr_critic) for i in
                                 range(self.n_agents)]  # 创建critic的Adam优化器

        # 初始化Replay buffer
        self.minimal_size = minimal_size  # 设置开始训练所需的最小buffer大小
        self.rep_buffer = {'state': [deque(maxlen=buffer_size) for _ in range(self.n_agents)],  # 为每个agent创建状态buffer
                           'action': [deque(maxlen=buffer_size) for _ in range(self.n_agents)],  # 为每个agent创建动作buffer
                           'reward': [deque(maxlen=buffer_size) for _ in range(self.n_agents)],  # 创建奖励buffer
                           'next_state': [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
                           # 为每个agent创建下一状态buffer
                           'done': [deque(maxlen=buffer_size) for _ in range(self.n_agents)]}  # 创建完成标志buffer

        # 设置折扣因子
        self.gamma = torch.Tensor([gamma]).cuda(device=device)  # 创建折扣因子张量并移至GPU

        # 设置初始探索率
        self.explore = explore  # 设置初始探索率

        # 设置探索率衰减参数
        self.explore_decay = explore_decay  # 设置探索率衰减系数
        self.explore_decay_interval = explore_decay_interval  # 设置探索率衰减间隔
        self.step = 1  # 初始化步数计数器
        self.train_step = 1  # 初始化训练步数计数器
        self.train_interval = train_interval  # 设置训练间隔

    # 往replay buffer中添加一个trajectory，
    def add_replay_buffer(self, state, action, reward, next_state, done):
        # 添加trajectory到replay buffer
        for i in range(self.n_agents):
            for CR in range(CR_Total):
                self.rep_buffer['state'][i].append(state[i][CR])  # 添加每个agent的状态
                self.rep_buffer['action'][i].append([action[i][CR]])  # 添加每个agent的动作
                self.rep_buffer['next_state'][i].append(next_state[i][CR])  # 添加每个agent的下一个状态
                self.rep_buffer['reward'][i].append(reward[i][CR])  # 添加奖励
                self.rep_buffer['done'][i].append(done[i][CR])  # 添加是否结束的标志

    def train(self):
        # 如果buffer的大小大于等于minimal_size则开始DDPG网络的更新
        # 如果replay buffer中的样本数量达到最小训练要求，则开始训练
        if (len(self.rep_buffer['state'][0]) >= self.minimal_size):
            self.explore = max(self.explore * self.explore_decay, 0.01)  # 更新探索率，但不低于0.01
            if self.train_step % self.train_interval == 0:  # 每train_interval步训练一次
                for step in range(DRL_steps):  # 执行DRL_steps次训练
                    # 从replay buffer中随机采样一个batch
                    indices = list(range(len(self.rep_buffer['state'][0])))
                    random.shuffle(indices)  # 打乱索引
                    # 根据打乱的索引获取batch数据
                    state = [[self.rep_buffer['state'][i][index] for index in indices[:self.bs]] for i in
                             range(self.n_agents)]
                    action = [[self.rep_buffer['action'][i][index] for index in indices[:self.bs]] for i in
                              range(self.n_agents)]
                    reward = [[self.rep_buffer['reward'][i][index] for index in indices[:self.bs]] for i in
                              range(self.n_agents)]
                    next_state = [[self.rep_buffer['next_state'][i][index] for index in indices[:self.bs]] for i in
                                  range(self.n_agents)]
                    done = [[self.rep_buffer['done'][i][index] for index in indices[:self.bs]] for i in
                            range(self.n_agents)]

                    # 将数据转换为tensor并移到GPU
                    reward = torch.Tensor(reward).cuda(device=self.device)
                    done = torch.Tensor(done).cuda(device=self.device)

                    action = [[torch.tensor(a) for a in a_list] for a_list in action]
                    action = [torch.stack(a) for a in action]
                    action = torch.cat(action, dim=1).cuda(device=self.device)

                    s_12 = [state[0][idx][:2] for idx in range(len(state[0]))]
                    s_12 = [torch.tensor(s).cuda(device=self.device) for s in s_12]
                    s_12 = torch.stack(s_12)

                    s_remain = [[s[2:] for s in s_list] for s_list in state]
                    s_remain = [[torch.tensor(s).cuda(device=self.device) for s in s_list] for s_list in s_remain]
                    s_remain = [torch.stack(s) for s in s_remain]
                    s_remain_tensor = torch.cat(s_remain, dim=1)
                    multi_state = torch.cat([s_12, s_remain_tensor], dim=1).cuda(device=self.device)
                    state = [[torch.tensor(s).cuda(device=self.device) for s in s_list] for s_list in state]
                    state = [torch.stack(s) for s in state]

                    ns_12 = [next_state[0][idx][:2] for idx in range(len(next_state[0]))]
                    ns_12 = [torch.tensor(s).cuda(device=self.device) for s in ns_12]
                    ns_12 = torch.stack(ns_12)
                    ns_remain = [[s[2:] for s in s_list] for s_list in next_state]
                    ns_remain = [[torch.tensor(s).cuda(device=self.device) for s in s_list] for s_list in ns_remain]
                    ns_remain = [torch.stack(s) for s in ns_remain]
                    ns_remain_tensor = torch.cat(ns_remain, dim=1)
                    multi_next_state = torch.cat([ns_12, ns_remain_tensor], dim=1).cuda(device=self.device)

                    next_state = [[torch.tensor(ns).cuda(device=self.device) for ns in ns_list] for ns_list in
                                  next_state]
                    next_state = [torch.stack(ns) for ns in next_state]

                    for i in range(self.n_agents):
                        # 计算目标Q值
                        with torch.no_grad():  # 不计算梯度
                            next_action = [self.actors_target[i](next_state[i].float()) for i in
                                           range(self.n_agents)]  # 使用目标actor网络预测下一个动作
                            next_action_tensor = torch.cat(next_action, dim=1).cuda(device=self.device)
                            Q_next = self.critics_target[i](
                                torch.cat([multi_next_state.float(), next_action_tensor], dim=1))

                            Q_target = reward[i].unsqueeze(1) + self.gamma * Q_next * (1.0 - done[i].unsqueeze(1))

                            # print(Q_target.shape)

                        # 更新critic网络
                        self.optimizer_critic[i].zero_grad()  # 清空梯度
                        Q_losses = self.critics[i](torch.cat([multi_state.float(), action.float()], dim=1))
                        # print(Q_losses.shape)
                        Q_loss = self.criterion_critic(Q_losses, Q_target.detach())  # 计算critic损失
                        Q_loss.backward()  # 反向传播
                        self.optimizer_critic[i].step()  # 更新critic网络参数

                        # 更新actor网络
                        # for i in range(self.n_agents):
                        self.optimizer_actor[i].zero_grad()  # 清空梯度
                        self.optimizer_critic[i].zero_grad()  # 清空梯度
                        multi_action = [self.actors[idx](state[idx].float()) for idx in range(self.n_agents)]
                        multi_action_tensor = torch.cat(multi_action, dim=1).cuda(device=self.device)
                        A_loss = -self.critics[i](
                            torch.cat([multi_state.float(), multi_action_tensor], dim=1))  # 使用actor网络生成动作，然后用critic网络评估
                        A_loss = A_loss.mean()
                        A_loss.backward()  # 反向传播
                        self.optimizer_actor[i].step()  # 更新actor网络参数

                    self.step += 1  # 增加步数计数

                    # 更新目标网络
                    for idex in range(self.n_agents):
                        for name, p in self.actors_target[idex].named_parameters():
                            p.data = (1 - self.lr_actor_target) * p.data + self.lr_actor_target * \
                                     self.actors[idex].state_dict()[
                                         name].data.clone()  # 软更新actor目标网络

                        for name, p in self.critics_target[idex].named_parameters():
                            p.data = (1 - self.lr_critic_target) * p.data + self.lr_critic_target * \
                                     self.critics[idex].state_dict()[
                                         name].data.clone()  # 软更新critic目标网络

            self.train_step += 1  # 增加训练步数计数

    # 无噪声地输出一个action，用于测试
    def take_action(self, state, idx):
        with torch.no_grad():  # 禁用梯度计算，减少内存使用并加速计算
            action = self.actors[idx](torch.Tensor(state).cuda(
                device=self.device)).cpu().numpy()  # 将状态转换为tensor，送入GPU，通过actor网络计算动作，然后转回CPU并转为numpy数组
            return action  # 返回计算得到的动作

    # 有噪声地输出一个action，用于训练
    def take_action_with_noise(self, state, idx):
        with torch.no_grad():  # 禁用梯度计算

            action = self.actors[idx](torch.Tensor(state).cuda(device=self.device)).cpu().numpy()  # 同上，计算无噪声的动作
            # assert not np.isnan(action).any(), "Actor 输出 NaN！"
            explore = self.explore  # 获取当前的探索参数
            action_noise = action + np.random.normal(loc=0.0, scale=explore, size=self.dim_action)  # 给动作添加高斯噪声
            for i in range(len(action_noise)):  # 对每个动作维度进行处理
                local_explore = explore  # 初始化局部探索参数
                while action_noise[i] < 0.0 or action_noise[i] > 1.0:  # 如果动作超出[0,1]范围
                    local_explore /= 2  # 将局部探索参数减半
                    if local_explore <= 0.0001:  # 如果局部探索参数太小
                        action_noise[i] = action[i]  # 直接使用原始动作
                        break
                    action_noise[i] = np.random.normal(loc=action_noise[i], scale=local_explore, size=1)  # 重新生成噪声
                # print(action_noise[0])
            return action, action_noise  # 返回原始动作和带噪声的动作

    # 保存模型
    def save(self, episode):
        save_dir = './DRL_Nets6'
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), os.path.join(save_dir, 'actor_' + str(i)
                                                                 + '_' + str(episode) + '.pth'))
            torch.save(self.actors_target[i].state_dict(), os.path.join(save_dir, 'actor_target_' + str(i)
                                                                        + '_' + str(episode) + '.pth'))
            torch.save(self.critics[i].state_dict(),
                       os.path.join(save_dir, 'critic_' + str(i) + '_' + str(episode) + '.pth'))  # 保存critic网络的参数
            torch.save(self.critics_target[i].state_dict(),
                       os.path.join(save_dir,
                                    'critic_target_' + str(i) + '_' + str(episode) + '.pth'))  # 保存目标critic网络的参数

    # 加载模型
    def load(self, episode):
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(
                torch.load('./DRL_Nets6/actor_' + str(i) + '_' + str(episode) + '.pth', map_location=self.device))
            self.actors_target[i].load_state_dict(
                torch.load('./DRL_Nets6/actor_target_' + str(i) + '_' + str(episode) + '.pth',
                           map_location=self.device))
            self.critics[i].load_state_dict(
                torch.load('./DRL_Nets6/critic_' + str(i) + '_' + str(episode) + '.pth', map_location=self.device))
            self.critics_target[i].load_state_dict(
                torch.load('./DRL_Nets6/critic_target_' + str(i) + '_' + str(episode) + '.pth',
                           map_location=self.device))


# 联邦学习用户的训练进程(神经网络训练)
class Training(mp.Process):
    def __init__(self,
                 C=4,  # clipping bound
                 bs=batch_size,  # 本地更新的batch size
                 lr=0.1,  # 学习率
                 el=epoch_local,  # 一轮本地更新的epoch数
                 ins=mp.Queue(),  # 输入队列，用于进程间通信
                 outs=mp.Queue(),  # 输出队列，用于进程间通信
                 device='cuda:0'  # 选择的gpu设备
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
                 device='cuda:0'  # 选择的gpu设备
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
        # rho_per_epoch = 2 / (sigma ** 2)  # 计算每个epoch的隐私预算消耗
        # self.rho_used = self.el * rho_per_epoch  # 计算本次训练使用的总隐私预算
        # self.rho_used = sigma2rho(sigma, self.bs, L)
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
                 device='cuda:0'  # 选择的gpu设备
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
    agent = Agent(n_agents=N,  # 初始化Agent对象，设置智能体数量为N
                  dim_state=5,  # 设置状态空间维度为5
                  dim_action=1)  # 设置动作空间维度为1
    # agent.load(999)
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
        trainset = CUDAMNIST(root='./data', train=True,  # 为每个CUDA设备创建一个MNIST训练集
                             download=True, pre_transform=transform, device='cuda:' + str(i))
        trainsets.append(trainset)  # 将创建的训练集添加到列表中

    indices = list(range(len(trainsets[0])))  # 创建数据集索引列表
    random.shuffle(indices)  # 随机打乱索引顺序
    dataset_num = int(len(trainsets[0]) / N)  # 计算每个客户端分配的数据量
    trsets = []  # 初始化客户端训练集列表
    for i in range(N):  # 遍历所有客户端
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
        print('Explore ', agent.explore)  # 打印当前探索率
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

        states = [[] for _ in range(N)]
        actions = [[] for _ in range(N)]
        rewards = [[] for _ in range(N)]
        next_states = [[] for _ in range(N)]
        done = [[] for _ in range(N)]
        last_reward = [0.0 for _ in range(N)]  # 最后一轮的reward

        isDone = 0.0  # 初始化完成标志为0
        for CR in range(CR_Total):  # 遍历每一轮通信
            if CR == CR_Total - 1:  # 如果是最后一轮
                isDone = 1.0  # 设置完成标志为1
            sigma_avg = 0  # 初始化平均sigma值
            avgloss = []
            flag = -1
            for i, client in enumerate(clients):  # 遍历每个客户端
                avgloss.append(sum(losses[i])/len(losses[i]))
                # if CR > 1:
                #     avgloss.append(0.5 * losses[i][CR] + 0.3 * losses[i][CR - 1] + 0.2 * losses[i][CR - 2])
                # elif CR == 0:
                #     avgloss.append(losses[i][CR])
                # elif CR == 1:
                #     avgloss.append(0.5 * losses[i][CR] + 0.5 * losses[i][CR - 1])

                state = [CR, CR_Total - CR, client.rho_total - client.rho, client.rho, avgloss[i]]

                client.state = state  # 设置客户端状态
                action, action_noise = agent.take_action_with_noise(state=state, idx=i)  # 获取带噪声的动作

                rho_used = action_noise[0] * (rho_used_max - rho_used_min) + rho_used_min  # 将动作映射到sigma值

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
                losses[i].append(avg_loss)
                global_loss += avg_loss  # 累加损失
                global_acc += acc  # 累加准确率
                client.local_acc = acc  # 设置客户端的本地准确率
            global_loss /= N  # 计算平均全局损失
            global_acc /= N  # 计算平均全局准确率

            for i, client in enumerate(clients):
                states[i].append(client.state)
                actions[i].append(client.action)
                rewards[i].append(2 * (avgloss[i] - losses[i][CR + 1] + rho_useds[i][CR] - rho_useds[i][CR - 1]))
                # print(f"avg:{rho_useds[i][CR]},loss:{rho_useds[i][CR-1]}")
                next_states[i].append([CR + 1, CR_Total - (CR + 1), client.rho_total - client.rho, client.rho,
                                       sum(losses[i])/len(losses[i])])
                done[i].append(isDone)  # 设置完成标志

            if CR == CR_Total - 1:
                for i, client in enumerate(clients):
                    if client.rho < 0:
                        last_reward[i] = abs(client.rho)
                    else:
                        last_reward[i] = abs(client.rho)
                    # print(last_reward[i])
                # if CR == 19:
                #     if client.rho < 0:
                #         rewards.append(avgloss[i] - losses[i][CR + 1] - 5 * abs(client.rho))
                #     else:
                #         rewards.append(avgloss[i] - losses[i][CR + 1] - 3 * abs(client.rho))
                # else:
                #     if client.rho - (CR_Total - CR - 2) * rho_useds[i][CR] < rho_used_min:
                #         exceed = 5 * abs(client.rho)
                #     rewards.append(avgloss[i] - losses[i][CR + 1] - exceed)

            global_accs.append(global_acc)  # 记录全局准确率
            last_global_acc = global_acc  # 更新上一轮的全局准确率
            global_losses.append(global_loss)
            last_global_loss = global_loss

            agent.train()  # 训练DRL代理

            # if flag != -1:
            #     break

        # reward 从最后一轮向前传递
        for i in range(N):
            for CR in range(CR_Total):
                rewards[i][CR] = rewards[i][CR] - last_reward[i]

        agent.add_replay_buffer(state=states, action=actions, reward=rewards, next_state=next_states, done=done)

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

            flag = -1

            for CR in range(CR_Total):  # 遍历每一轮通信
                for i, client in enumerate(clients):  # 遍历每个客户端
                    avgloss = sum(losses[i])/len(losses[i])

                    # if CR > 1:
                    #     avgloss = 0.5 * losses_DRL[i][CR] + 0.3 * losses_DRL[i][CR - 1] + 0.2 * losses_DRL[i][CR - 2]
                    # elif CR == 0:
                    #     avgloss = losses_DRL[i][CR]
                    # elif CR == 1:
                    #     avgloss = 0.5 * losses_DRL[i][CR] + 0.5 * losses_DRL[i][CR - 1]
                    state = [CR, CR_Total - CR, client.rho_total - client.rho, client.rho, avgloss]
                    action = agent.take_action(state=state, idx=i)  # 根据当前状态，让agent选择一个动作
                    rho_used = action[0] * (rho_used_max - rho_used_min) + rho_used_min  # 将动作转换为实际的sigma值
                    # rho_used = action[0] * client.rho
                    # if client.rho <= rho_used_min:
                    #     flag = i
                    # print(f"{rho_used}={action[0]}*{client.rho}")
                    # rho_used = np.clip(rho_used,rho_used_min,rho_used_max)
                    # if CR == 19 and client.rho > 0.0:
                    #     rho_used = np.clip(client.rho, 0.0, rho_used_max)
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

                # if flag != -1:
                #     break

            print('Original sigmas:',
                  [np.mean([sigmas[i][CR] for i in range(N)]) for CR in range(len(sigmas[i]))])  # 打印每轮通信中所有客户端的平均sigma值
            rho_used_sums = [sum(rho_useds[i]) for i in range(N)]  # 计算每个客户端使用的总rho值

            returns_DRL = [global_accs_DRL[-1] - global_accs_DRL[0]]  # 计算DRL方法的回报（最终准确率与初始准确率的差）
            final_accs.append(global_accs_DRL[-1])  # 记录最终的全局准确率
            returns.append(returns_DRL)  # 记录回报

            low_index = max(len(final_accs) - 10, 0)  # 计算最近10个准确率的起始索引
            mean_acc = np.mean(final_accs[low_index:])  # 计算最近10个准确率的平均值
            mean_final_accs.append(mean_acc)  # 记录平均准确率
            # 如果平均准确率超过历史最高，并且已经训练了超过10个episode，则保存模型
            if (mean_acc > high_acc and episode > 10) or episode > 990 or episode % 100 == 0:
                if mean_acc > high_acc:
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
