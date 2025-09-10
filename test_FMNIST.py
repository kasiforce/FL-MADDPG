from collections.abc import Callable, Iterable, Mapping
from typing import Any
import numpy as np
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
import os
import copy
import warnings
import matplotlib.pyplot as plt
import matplotlib
import json

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

L = 103
N = 10  # Number of clients
batch_size = 256
episodes = 10  # 测试的episode数
CR_Total = 20
his_len = 3
rho_min = 2.0
rho_max = 6.0
epoch_local = 1
DRL_steps = 5
loss_max = -np.log(0.1)
sigma_max = 2.0
sigma_min = 1.0
rho_used_min = 2 * (L ** 2) / ((batch_size ** 2) * (sigma_max ** 2))
rho_used_max = 2 * (L ** 2) / ((batch_size ** 2) * (sigma_min ** 2))


# 根据公式，rho和sigma之间相互转化
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


class CUDAMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 pre_transform: Callable[..., Any] = None,
                 transform: Callable[..., Any] = None,
                 target_transform: Callable[..., Any] = None,
                 download: bool = False,
                 device: str = 'cuda:0') -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.type(torch.FloatTensor)
        for i in range(len(self)):
            self.data[i] = pre_transform((self.data[i] / 255.0).numpy())
            self.targets[i] = torch.Tensor([self.targets[i]]).squeeze_().long()
        self.data = self.data.unsqueeze_(1).cuda(device=device)
        self.targets = self.targets.cuda(device=device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CUDAMNIST_Test(torchvision.datasets.FashionMNIST):
    def __init__(self,
                 root: str,
                 train: bool = False,
                 pre_transform: Callable[..., Any] = None,
                 transform: Callable[..., Any] = None,
                 target_transform: Callable[..., Any] = None,
                 download: bool = False,
                 device: str = 'cuda:0') -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.type(torch.FloatTensor)
        for i in range(len(self)):
            self.data[i] = pre_transform((self.data[i] / 255.0).numpy())
            self.targets[i] = torch.Tensor([self.targets[i]]).squeeze_().long()
        self.data = self.data.unsqueeze_(1).cuda(device=device)
        self.targets = self.targets.cuda(device=device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


class Critic(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state + dim_action, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
    # 并且如果buffer的大小大于等于minimal_size则开始DDPG网络的更新
    def train(self, state, action, reward, next_state, done):
        # 添加trajectory到replay buffer
        for i in range(self.n_agents):
            self.rep_buffer['state'][i].append(state[i])  # 添加每个agent的状态
            self.rep_buffer['action'][i].append([action[i]])  # 添加每个agent的动作
            self.rep_buffer['next_state'][i].append(next_state[i])  # 添加每个agent的下一个状态
            self.rep_buffer['reward'][i].append(reward[i])  # 添加奖励
            self.rep_buffer['done'][i].append(done[i])  # 添加是否结束的标志

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

                            # print("done  ",done[i].shape)
                            # reward[i] = reward[i].unsqueeze(1)
                            # print("reward  ",reward[i].shape)
                            # print("Q_next   ",Q_next.shape)
                            # Q_target = reward[i] + self.gamma * Q_next * (1.0 - done[i])  # 计算目标Q值
                            Q_target = reward[i].unsqueeze(1) + self.gamma * Q_next * (1.0 - done[i].unsqueeze(1))

                            # print(Q_target.shape)

                        # 更新critic网络
                        self.optimizer_critic[i].zero_grad()  # 清空梯度
                        Q_losses = self.critics[i](torch.cat([multi_state.float(), action.float()], dim=1))
                        # print(Q_losses.shape)
                        Q_loss = torch.mean(self.criterion_critic(Q_losses, Q_target.detach()))  # 计算critic损失
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
                        # 当探索幅度过小时，强制设置一个微小偏移量（避免等于 0 或 1）
                        # if action_noise[i] <= 0.0:
                        #     action_noise[i] = 1e-6  # 避免等于 0.0
                        # elif action_noise[i] >= 1.0:
                        #     action_noise[i] = 1.0 - 1e-6  # 避免等于 1.0
                        break
                    action_noise[i] = np.random.normal(loc=action_noise[i], scale=local_explore, size=1)  # 重新生成噪声
                # print(action_noise[0])
            return action, action_noise  # 返回原始动作和带噪声的动作

    # 保存模型
    def save(self, episode):
        save_dir = './DRL_NetsF'
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
                torch.load('./DRL_NetsF/actor_' + str(i) + '_' + str(episode) + '.pth', map_location=self.device))
            self.actors_target[i].load_state_dict(
                torch.load('./DRL_NetsF/actor_target_' + str(i) + '_' + str(episode) + '.pth',
                           map_location=self.device))
            self.critics[i].load_state_dict(
                torch.load('./DRL_NetsF/critic_' + str(i) + '_' + str(episode) + '.pth', map_location=self.device))
            self.critics_target[i].load_state_dict(
                torch.load('./DRL_NetsF/critic_target_' + str(i) + '_' + str(episode) + '.pth',
                           map_location=self.device))


class Training(mp.Process):
    def __init__(self, C=4, bs=batch_size, lr=0.1, el=epoch_local, ins=mp.Queue(),
                 outs=mp.Queue(), device='cuda:0') -> None:
        mp.Process.__init__(self)

        # Client parameters
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

    def run(self):
        # Initiate local state
        self.trset = self.ins.get()
        self.trainloader = torch.utils.data.DataLoader(self.trset, batch_size=self.bs,
                                                       shuffle=True, num_workers=0, pin_memory=False)
        self.net = Net()
        pe = opacus.PrivacyEngine()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        self.net, self.optimizer, _ = pe.make_private(
            module=self.net,
            data_loader=self.trainloader,
            optimizer=self.optimizer,
            noise_multiplier=1.0,
            max_grad_norm=self.C
        )
        self.net.cuda(device=self.device)
        print('Dataset loaded successfully of', self.device)
        self.outs.put(1)

        while True:
            mode, net, sigma = self.ins.get()
            if mode == 0:  # Terminate mode
                break
            elif mode == 1:  # training mode
                self.net.load_state_dict(net.state_dict())
                self.optimizer.noise_multiplier = sigma
                for epoch in range(self.el):
                    for i, data in enumerate(self.trainloader):
                        self.optimizer.zero_grad()
                        inputs, labels = data
                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                self.outs.put(self.net)

            # elif mode == 1:  # 训练模式
            #     self.net.load_state_dict(net.state_dict(), strict=False)
            #     for epoch in range(self.el):
            #         for i, data in enumerate(self.trainloader):
            #             self.optimizer.zero_grad()
            #             inputs, labels = data
            #             # inputs = inputs.cuda(device=self.device)
            #             # labels = labels.cuda(device=self.device)
            #             outputs = self.net(inputs)
            #             loss = self.criterion(outputs, labels)
            #             loss.backward()

            #             # 梯度扰动
            #             for param in self.net.parameters():
            #                 noise = torch.normal(0, sigma * self.C, param.grad.shape, device=self.device)
            #                 param.grad += noise

            #             self.optimizer.step()
            #     self.outs.put(self.net)

            else:  # Testing mode
                self.net.load_state_dict(net.state_dict())
                with torch.no_grad():
                    correct = 0
                    total = 0
                    avg_loss = 0
                    count = 0
                    for i, data in enumerate(self.trainloader):
                        inputs, labels = data
                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        avg_loss = avg_loss * (count / (count + 1)) + loss.item() / (count + 1)
                        count += 1
                    acc = correct / total
                    self.outs.put([acc, avg_loss])


class Client():
    def __init__(self, id=-1, C=4, rho_total=20.0,
                 bs=batch_size, trset=None, lr=0.1, el=epoch_local, hl=his_len, agent=None, device='cuda:0') -> None:
        # Basic client status
        self.id = id
        self.rho = rho_total
        self.rho_total = rho_total
        self.C = C
        self.bs = bs
        self.trset = trset
        self.lr = lr
        self.el = el
        self.hl = hl
        self.sigma = sigma_max
        self.rho_used = 2 * self.el / (self.sigma ** 2)
        self.last_local_acc = 0.1
        self.local_acc = 0.1
        self.isAvailable = True

        # DRL variables
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = None
        self.real_action = None
        self.acc = 0
        self.device = device

        self.agent = agent

        # Training process
        self.ins = mp.Queue(1)
        self.outs = mp.Queue(1)
        self.process = Training(C=self.C, bs=self.bs, lr=self.lr, el=self.el,
                                ins=self.ins, outs=self.outs, device=device)
        self.process.start()
        self.ins.put(self.trset)
        self.outs.get()  # Waiting for data loading

    def reset(self, rho_total):
        self.rho = rho_total
        self.rho_total = rho_total
        self.action = None
        self.sigma = sigma_max
        self.isAvailable = True
        self.last_local_acc = 0.1
        self.local_acc = 0.1

    def train(self, net, sigma, L):
        self.sigma = sigma
        # rho_per_epoch = 2 / (sigma ** 2)
        # self.rho_used = self.el * rho_per_epoch
        self.rho_used = sigma2rho(sigma, self.bs, L)
        self.rho -= self.rho_used
        self.ins.put([1, net, sigma])

    def test(self, net):
        self.ins.put([2, net, 0])

    def terminate_process(self):
        self.ins.put([0, 0, 0])


class Server():
    def __init__(self, net, device='cuda:0') -> None:
        self.device = device
        self.net = Net()
        self.net.load_state_dict(net.state_dict())
        dummyOptimizer = optim.SGD(self.net.parameters(), lr=0)
        dummySet = [[0, 0]]
        dummyLoader = torch.utils.data.DataLoader(dummySet)
        privacyEngine = opacus.PrivacyEngine()
        self.net, _, _ = privacyEngine.make_private(
            module=self.net,
            optimizer=dummyOptimizer,
            data_loader=dummyLoader,
            noise_multiplier=4,
            max_grad_norm=4
        )
        self.dummy_net = copy.deepcopy(self.net)
        self.dummy_net.cuda(device=self.device)
        self.net.cuda(device=self.device)

    def reset(self, net):
        self.net = Net()
        self.net.load_state_dict(net.state_dict())
        dummyOptimizer = optim.SGD(self.net.parameters(), lr=0)
        dummySet = [[0, 0]]
        dummyLoader = torch.utils.data.DataLoader(dummySet)
        privacyEngine = opacus.PrivacyEngine()
        self.net, _, _ = privacyEngine.make_private(
            module=self.net,
            optimizer=dummyOptimizer,
            data_loader=dummyLoader,
            noise_multiplier=4,
            max_grad_norm=4
        )
        self.net.cuda(device=self.device)

    def aggregate(self, net_list):
        count = 0
        for net in net_list:
            self.dummy_net.load_state_dict(net.state_dict())
            for name, p in self.net.named_parameters():
                p.data = p.data * (count / (count + 1)) + self.dummy_net.state_dict()[name].data / (count + 1)
            count += 1


def test(net, testloader, device):
    correct = 0
    total = 0
    total_loss = 0.0  # 初始化总损失
    criterion = nn.CrossEntropyLoss()  # 定义损失函数

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)  # 使用 to(device) 代替 cuda(device=device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()  # 累加损失

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_loss = total_loss / len(testloader)  # 计算平均损失

    return accuracy, average_loss


if __name__ == '__main__':
    t0 = time.time()
    agent = Agent(n_agents=N,
                  dim_state=5,
                  dim_action=1)
    agent.load(episode=135)
    mp.set_start_method('spawn', force=True)

    # Initialize server
    net = Net()
    server = Server(net=net)

    # Split Datasets
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
    trainsets = []
    for i in range(torch.cuda.device_count()):
        trainset = CUDAMNIST(root='./data', train=True,
                             download=True, pre_transform=transform, device='cuda:' + str(i))
        trainsets.append(trainset)

    indices = list(range(len(trainsets[0])))
    random.shuffle(indices)
    dataset_num = int(len(trainsets[0]) / N)
    trsets = []
    for i in range(N):
        device_index = i % torch.cuda.device_count()
        device = 'cuda:' + str(device_index)
        trset = torch.utils.data.Subset(trainsets[device_index], indices[i * dataset_num: (i + 1) * dataset_num])
        trsets.append([trset, device])

    testset = CUDAMNIST_Test(root='./data', train=False,
                             download=True, pre_transform=transform, device='cuda:0')

    testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                             shuffle=True, num_workers=0, pin_memory=False)

    # Initialize clients
    clients = [Client(id=i, rho_total=np.random.uniform(low=rho_min, high=rho_max),
                      trset=trsets[i][0], agent=agent, device=trsets[i][1]) for i in range(N)]

    # Lists to store results from all episodes
    all_accs1 = []  # DRL
    all_accs2 = []  # Uniform
    all_accs4 = []  # Exp
    all_accs5 = []  # Linear
    all_rhos1 = []
    all_rhos2 = []
    all_rhos4 = []
    all_rhos5 = []
    all_sigmas1 = []
    all_sigmas2 = []
    all_sigmas4 = []
    all_sigmas5 = []
    all_rho_totals = []  # Store initial rho_total values for each episode

    # Lists to store accuracies for current episode

    # Start training
    episode = 0
    while episode < episodes:
        t1 = time.time()
        print('Episode', episode)

        accs1 = []
        accs2 = []
        accs4 = []
        accs5 = []

        # Reset clients and server
        net_init = Net()
        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)
        rhos_init = []
        for i, client in enumerate(clients):
            rho = np.random.uniform(low=rho_min, high=rho_max)
            rhos_init.append(rho)
            client.reset(rho_total=rho)
        all_rho_totals.append(rhos_init)

        # DRL test
        print('DRL test==========================================')
        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)

        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        global_accs_DRL = []
        acc, loss = test(net=server.net, testloader=testloader, device=server.device)
        global_accs_DRL.append(acc)

        losses = [[] for _ in range(N)]
        for client in clients:
            client.test(server.net)
        for i, client in enumerate(clients):
            acc, avg_loss = client.outs.get()
            losses[i].append(avg_loss)

        sigmas = [[] for _ in range(N)]
        rho_useds = [[] for _ in range(N)]
        rhos = [[] for _ in range(N)]

        for CR in range(CR_Total):
            isDone = 1.0 if CR == CR_Total - 1 else 0.0

            for i, client in enumerate(clients):  # 遍历每个客户端
                
                state = [CR, CR_Total - CR, client.rho_total - client.rho, client.rho, losses[i][CR]]
             
                client.state = state  # 设置客户端状态
                action, action_noise = agent.take_action_with_noise(state=state, idx=i)  # 获取带噪声的动作
              
                rho_used = action_noise[0] * (rho_used_max - rho_used_min) + rho_used_min  # 将动作映射到sigma值
            
                sigma = rho2sigma(rho_used, client.bs, L)

                sigmas[i].append(sigma)  # 将sigma值添加到列表
                rho_useds[i].append(rho_used)  # 将使用的隐私预算添加到列表
                client.sigma = sigmas[i][CR]  # 设置客户端的sigma值

                client.action = (rho_used - rho_used_min) / (rho_used_max - rho_used_min)
              

                client.train(net=server.net, sigma=client.sigma, rho_used=rho_used)

            net_list = []
            for client in clients:
                cnet = client.outs.get()
                net_list.append(cnet)
            server.aggregate(net_list=net_list)
            acc, _ = test(net=server.net, testloader=testloader, device=server.device)
            global_accs_DRL.append(acc)

            for client in clients:
                client.test(server.net)
            for i, client in enumerate(clients):
                acc, avg_loss = client.outs.get()
                losses[i].append(avg_loss)
       
        accs1.append(global_accs_DRL)
        # all_accs1.append(global_accs_DRL)
        all_rhos1.append(rho_useds)
        all_sigmas1.append(sigmas)

        # Uniform test
        # print('Uniform test==========================================')
        sigma_Uniform = []
        sigmas = [[] for _ in range(N)]
        rhos = [[] for _ in range(N)]
        for i, client in enumerate(clients):
            sigma = rho2sigma(rho=client.rho_total / CR_Total, bs=client.bs, L=L)
            sigma_Uniform.append(sigma)
            sigmas[i] = [sigma for _ in range(CR_Total)]

        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)

        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])
        global_accs_Uniform = []
        acc, _ = test(net=server.net, testloader=testloader, device=server.device)
        global_accs_Uniform.append(acc)

        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
                sigma = sigma_Uniform[i]
                client.sigma = sigma
                client.train(net=server.net, sigma=sigma, L=L)
            net_list = []
            for client in clients:
                cnet = client.outs.get()
                net_list.append(cnet)
            server.aggregate(net_list=net_list)
            acc, _ = test(net=server.net, testloader=testloader, device=server.device)
            global_accs_Uniform.append(acc)

        for i, client in enumerate(clients):
            rhos[i].append(client.rho)
        accs2.append(global_accs_Uniform)
        # all_accs2.append(global_accs_Uniform)
        all_rhos2.append(rhos)
        all_sigmas2.append(sigmas)

        # Exp test
        # print('Exp test==========================================')
        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)
        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        sigmas = [[] for _ in range(N)]
        rho_useds = [[] for _ in range(N)]
        rhos = [[] for _ in range(N)]
        decay_ratio = 2 ** (1 / (CR_Total - 1))

        for i, client in enumerate(clients):
            rho_useds[i].append((1 - decay_ratio) * client.rho_total / (1 - decay_ratio ** CR_Total))
            for CR in range(CR_Total):
                rho_used = rho_useds[i][0] * (decay_ratio ** CR)
                sigma = rho2sigma(rho=rho_used, bs=client.bs, L=L)
                sigmas[i].append(sigma)

        global_accs_Exp = []
        acc, _ = test(net=server.net, testloader=testloader, device=server.device)
        global_accs_Exp.append(acc)

        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
                sigma = sigmas[i][CR]
                client.sigma = sigma
                client.train(net=server.net, sigma=sigma, L=L)
            net_list = []
            for client in clients:
                cnet = client.outs.get()
                net_list.append(cnet)
            server.aggregate(net_list=net_list)
            acc, _ = test(net=server.net, testloader=testloader, device=server.device)
            global_accs_Exp.append(acc)

        for i, client in enumerate(clients):
            rhos[i].append(client.rho)
        accs4.append(global_accs_Exp)
        # all_accs4.append(global_accs_Exp)
        all_rhos4.append(rhos)
        all_sigmas4.append(sigmas)

        # Linear test
        # print('Linear test==========================================')
        net = Net()
        net.load_state_dict(net_init.state_dict())
        server.reset(net=net)
        for i, client in enumerate(clients):
            client.reset(rho_total=rhos_init[i])

        sigmas = [[] for _ in range(N)]
        rho_useds = [[] for _ in range(N)]
        rhos = [[] for _ in range(N)]
        decays = [2 * client.rho_total / (CR_Total * (CR_Total - 1) * 3) for client in clients]

        for i, client in enumerate(clients):
            rho_useds[i].append(2 / 3 * (client.rho_total / CR_Total))
            for CR in range(CR_Total):
                rho_used = rho_useds[i][0] + decays[i] * CR
                sigma = rho2sigma(rho=rho_used, bs=client.bs, L=L)
                sigmas[i].append(sigma)

        global_accs_Linear = []
        acc, _ = test(net=server.net, testloader=testloader, device=server.device)
        global_accs_Linear.append(acc)

        for CR in range(CR_Total):
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
                sigma = sigmas[i][CR]
                client.sigma = sigma
                client.train(net=server.net, sigma=sigma, L=L)
            net_list = []
            for client in clients:
                cnet = client.outs.get()
                net_list.append(cnet)
            server.aggregate(net_list=net_list)
            acc, _ = test(net=server.net, testloader=testloader, device=server.device)
            global_accs_Linear.append(acc)

        for i, client in enumerate(clients):
            rhos[i].append(client.rho)
        accs5.append(global_accs_Linear)
        # all_accs5.append(global_accs_Linear)
        all_rhos5.append(rhos)
        all_sigmas5.append(sigmas)

        # Print average accuracies for this episode
        avg_accs1 = [np.mean([l[CR] for l in accs1]) for CR in range(CR_Total + 1)]
        all_accs1.append(avg_accs1)
        avg_accs2 = [np.mean([l[CR] for l in accs2]) for CR in range(CR_Total + 1)]
        all_accs2.append(avg_accs2)
        avg_accs4 = [np.mean([l[CR] for l in accs4]) for CR in range(CR_Total + 1)]
        all_accs4.append((avg_accs4))
        avg_accs5 = [np.mean([l[CR] for l in accs5]) for CR in range(CR_Total + 1)]
        all_accs5.append(avg_accs5)
        print('acc DRL:', avg_accs1[-1])
        print('acc Uniform:', avg_accs2[-1])
        print('acc Exp:', avg_accs4[-1])
        print('acc Arithmetic:', avg_accs5[-1])

        print('Time spent for this episode:', time.time() - t1)
        episode += 1

    # Calculate averages across all episodes
    avg_accs1 = np.mean(all_accs1, axis=0)
    avg_accs2 = np.mean(all_accs2, axis=0)
    avg_accs4 = np.mean(all_accs4, axis=0)
    avg_accs5 = np.mean(all_accs5, axis=0)

    avg_rhos1 = np.mean(all_rhos1, axis=0)
    avg_rhos2 = np.mean(all_rhos2, axis=0)
    avg_rhos4 = np.mean(all_rhos4, axis=0)
    avg_rhos5 = np.mean(all_rhos5, axis=0)

    avg_sigmas1 = np.mean(all_sigmas1, axis=0)
    avg_sigmas2 = np.mean(all_sigmas2, axis=0)
    avg_sigmas4 = np.mean(all_sigmas4, axis=0)
    avg_sigmas5 = np.mean(all_sigmas5, axis=0)

    # Calculate average rho_total for each client
    avg_rho_totals = np.mean(all_rho_totals, axis=0)

    # Plot average accuracies
    # plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(list(range(CR_Total + 1)), avg_accs1, linestyle='solid')
    plt.plot(list(range(CR_Total + 1)), avg_accs2, linestyle='dotted')
    plt.plot(list(range(CR_Total + 1)), avg_accs4, linestyle='dashed')
    plt.plot(list(range(CR_Total + 1)), avg_accs5, linestyle='dashdot')
    plt.xticks([0, 5, 10, 15, 20], size=12)
    plt.yticks(size=12)
    plt.xlabel('Communication Round $t$', fontsize=16)
    plt.ylabel('Average Test Accuracy', fontsize=16)
    plt.legend(['Our method', 'Uniform', 'Exponential', 'Arithmetic sequence'], fontsize=14)
    plt.grid()
    if not os.path.exists('./exp_data/'):
        os.makedirs('./exp_data/')
    plt.savefig('./exp_data/Avg_Acc_Test_MNIST.png')
    plt.close()

    # Plot average sigmas for representative clients
    # plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(list(range(CR_Total)), avg_sigmas1[0], linestyle='solid')
    plt.plot(list(range(CR_Total)), avg_sigmas1[1], linestyle='dotted')
    plt.plot(list(range(CR_Total)), avg_sigmas1[2], linestyle='dashed')
    plt.xticks([0, 5, 10, 15, 20], size=12)
    plt.yticks(size=12)
    plt.xlabel('Communication Round $t$', fontsize=16)
    plt.ylabel('Average Noise Level $\sigma_n^t$', fontsize=16)
    plt.legend(['Client with ' + r'$\rho_k=$' + "%.5f" % avg_rho_totals[0],
               'Client with ' + r'$\rho_k=$' + "%.5f" % avg_rho_totals[1],
               'Client with ' + r'$\rho_k=$' + "%.5f" % avg_rho_totals[2]], fontsize=14)
    plt.grid()
    plt.savefig('./exp_data/Avg_Sigmas_DRL_test_MNIST.png')
    plt.close()

    # Plot average rhos
    # plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(list(range(CR_Total)), avg_rhos1[0], linestyle='solid')
    plt.plot(list(range(CR_Total)), avg_rhos1[1], linestyle='dotted')
    plt.plot(list(range(CR_Total)), avg_rhos1[2], linestyle='dashed')
    plt.xticks([0, 5, 10, 15, 20], size=12)
    plt.yticks(size=12)
    plt.xlabel('Communication Round $t$', fontsize=16)
    plt.ylabel('Average Rho Level $\\rho_n^t$', fontsize=16)
    plt.legend(['Client with ' + r'$\rho_k=$' + "%.5f" % avg_rho_totals[0],
                'Client with ' + r'$\rho_k=$' + "%.5f" % avg_rho_totals[1],
                'Client with ' + r'$\rho_k=$' + "%.5f" % avg_rho_totals[2]], fontsize=14)
    plt.grid()
    plt.savefig('./exp_data/Avg_Rhos_DRL_test_MNIST.png')
    plt.close()

    # Save all experimental data
    exp_data = {
        'accs': [all_accs1, all_accs2, all_accs4, all_accs5],
        'rhos': [all_rhos1, all_rhos2, all_rhos4, all_rhos5],
        'sigmas': [all_sigmas1, all_sigmas2, all_sigmas4, all_sigmas5],
        'avg_accs': [avg_accs1.tolist(), avg_accs2.tolist(), avg_accs4.tolist(), avg_accs5.tolist()],
        'avg_rhos': [avg_rhos1.tolist(), avg_rhos2.tolist(), avg_rhos4.tolist(), avg_rhos5.tolist()],
        'avg_sigmas': [avg_sigmas1.tolist(), avg_sigmas2.tolist(), avg_sigmas4.tolist(), avg_sigmas5.tolist()]
    }

    with open("./exp_data/exp_data_MNIST.json", "w") as fp:
        json.dump(exp_data, fp)

    # Print final average accuracies
    print('Final average accuracies:')
    print('DRL:', avg_accs1[-1])
    print('Uniform:', avg_accs2[-1])
    print('Exp:', avg_accs4[-1])
    print('Linear:', avg_accs5[-1])

    # Terminate all clients
    for client in clients:
        client.terminate_process()
    for client in clients:
        client.process.join()

    print('Total time spent:', time.time() - t0)
