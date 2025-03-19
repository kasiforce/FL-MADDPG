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
episodes = 1  # 测试的episode数
CR_Total = 20
his_len = 3
rho_min = 2.0
rho_max = 6.0
epoch_local = 1
DRL_steps = 5
loss_max = -np.log(0.1)
sigma_max = 5.0
sigma_min = 0.5
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


class CUDAMNIST(torchvision.datasets.MNIST):
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


class CUDAMNIST_Test(torchvision.datasets.MNIST):
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
                 n_agents,
                 dim_state,
                 dim_action,
                 lr_actor=0.000015,
                 lr_critic=0.00015,
                 lr_actor_target=0.005,
                 lr_critic_target=0.005,
                 buffer_size=4000,
                 minimal_size=512,
                 gamma=0.998,
                 explore=1.0,
                 explore_decay=0.9996,
                 explore_decay_interval=1000,
                 bs=256,
                 train_interval=1,
                 device='cuda:0') -> None:

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

        # Network optimizer hyperparameters
        self.lr_actor_target = lr_actor_target
        self.lr_critic_target = lr_critic_target
        self.bs = bs
        self.lr_actor_target = lr_actor_target
        self.criterion_critic = nn.MSELoss()
        self.optimizer_actor = [optim.Adam(self.actors[i].parameters(), lr=lr_actor) for i in
                                range(self.n_agents)]  # 创建actor的Adam优化器

        self.optimizer_critic = [optim.Adam(self.critics[i].parameters(), lr=lr_critic) for i in
                                 range(self.n_agents)]  # 创建critic的Adam优化器
        # Replay buffer
        self.minimal_size = minimal_size
        self.rep_buffer = {'state': [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
                           'action': [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
                           'reward': [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
                           'next_state': [deque(maxlen=buffer_size) for _ in range(self.n_agents)],
                           'done': [deque(maxlen=buffer_size) for _ in range(self.n_agents)]}

        # Discount factor
        self.gamma = torch.Tensor([gamma]).cuda(device=device)

        # Initial explore rate
        self.explore = explore

        # Explore decay
        self.explore_decay = explore_decay
        self.explore_decay_interval = explore_decay_interval
        self.step = 1
        self.train_step = 1
        self.train_interval = train_interval

    def train(self, state, action, reward, next_state, done):
        for i in range(self.n_agents):
            self.rep_buffer['state'][i].append(state[i])
            self.rep_buffer['action'][i].append([action[i]])
            self.rep_buffer['next_state'][i].append(next_state[i])
            self.rep_buffer['reward'][i].append(reward[i])
            self.rep_buffer['done'][i].append(done[i])
        if (len(self.rep_buffer['state'][0]) >= self.minimal_size):
            self.explore = max(self.explore * self.explore_decay, 0.01)
            if self.train_step % self.train_interval == 0:
                for step in range(DRL_steps):
                    indices = list(range(len(self.rep_buffer['state'][0])))
                    random.shuffle(indices)
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

                    reward = torch.Tensor(reward).cuda(device=self.device)
                    done = torch.Tensor(done).cuda(device=self.device)

                    action = [[torch.tensor(a).cuda(device=self.device) for a in a_list] for a_list in action]
                    action = [torch.stack(a) for a in action]
                    action = torch.cat(action, dim=1).cuda(device=self.device)
                    s_12 = [state[0][idx][:2] for idx in range(len(state[0]))]
                    s_12 = [torch.tensor(s).cuda(device=self.device) for s in s_12]
                    s_12 = torch.stack(s_12)

                    # s_12 = torch.cat(s_12, dim=1)
                    s_remain = [[s[2:] for s in s_list] for s_list in state]
                    s_remain = [[torch.tensor(s).cuda(device=self.device) for s in s_list] for s_list in s_remain]
                    s_remain = [torch.stack(s) for s in s_remain]
                    s_remain_tensor = torch.cat(s_remain, dim=1)
                    # s_12 = s_12.unsqueeze(0).expand(s_remain_tensor.size(0), -1)
                    multi_state = torch.cat([s_12, s_remain_tensor], dim=1).cuda(device=self.device)
                    state = [[torch.tensor(s).cuda(device=self.device) for s in s_list] for s_list in state]
                    state = [torch.stack(s) for s in state]
                    ns_12 = [next_state[0][idx][:2] for idx in range(len(next_state[0]))]
                    ns_12 = [torch.tensor(s).cuda(device=self.device) for s in ns_12]
                    ns_12 = torch.stack(ns_12)
                    # ns_12 = [torch.tensor(s) for s in ns_12]
                    # ns_12 = torch.cat(ns_12, dim=1)
                    ns_remain = [[s[2:] for s in s_list] for s_list in next_state]
                    ns_remain = [[torch.tensor(s).cuda(device=self.device) for s in s_list] for s_list in ns_remain]
                    ns_remain = [torch.stack(s) for s in ns_remain]
                    ns_remain_tensor = torch.cat(ns_remain, dim=1)

                    next_state = [[torch.tensor(ns).cuda(device=self.device) for ns in ns_list] for ns_list in
                                  next_state]
                    next_state = [torch.stack(ns) for ns in next_state]
                    multi_next_state = torch.cat([ns_12, ns_remain_tensor], dim=1).cuda(device=self.device)

                    for i in range(self.n_agents):
                        with torch.no_grad():
                            next_action = [self.actors_target[i](next_state[i].float()) for i in
                                           range(self.n_agents)]  # 使用目标actor网络预测下一个动作
                            next_action_tensor = torch.cat(next_action, dim=1).cuda(device=self.device)
                            Q_next = self.critics_target[i](
                                torch.cat([multi_next_state.float(), next_action_tensor], dim=1))
                            Q_target = reward[i].unsqueeze(1) + self.gamma * Q_next * (
                                        1.0 - done[i].unsqueeze(1))  # 计算目标Q值

                        # Update Q-functions
                        self.optimizer_critic[i].zero_grad()  # 清空梯度
                        Q_losses = self.critics[i](torch.cat([multi_state.float(), action.float()], dim=1))
                        # Q_losses = []
                        # for i in range(self.n_agents):
                        #     Q_pred = self.critic(torch.cat([state[i], action[i]], dim=1))  # 使用critic网络预测当前状态-动作对的Q值
                        #     Q_losses.append(Q_pred)
                        Q_loss = self.criterion_critic(Q_losses, Q_target.detach())  # 计算critic损失
                        Q_loss.backward()  # 反向传播
                        self.optimizer_critic[i].step()  # 更新critic网络参数

                        # Update actor
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

            self.train_step += 1

    def take_action(self, state, i):
        with torch.no_grad():
            action = self.actors[i](torch.Tensor(state).cuda(device=self.device)).cpu().numpy()
            return action

    def take_action_with_noise(self, state, i):
        with torch.no_grad():
            action = self.actors[i](torch.Tensor(state).cuda(device=self.device)).cpu().numpy()
            explore = self.explore
            action_noise = action + np.random.normal(loc=0.0, scale=explore, size=self.dim_action)
            for i in range(len(action_noise)):
                local_explore = explore
                while action_noise[i] < 0.0 or action_noise[i] > 1.0:
                    local_explore /= 2
                    if local_explore <= 0.0001:
                        action_noise[i] = action[i]
                        break
                    action_noise[i] = np.random.normal(loc=action_noise[i], scale=local_explore, size=1)
            return action, action_noise

    def save(self, episode):
        save_dir = './DRL_Nets6'
        os.makedirs(save_dir, exist_ok=True)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), os.path.join(save_dir, 'actor_' + str(i)
                                                                 + '_' + str(episode) + '.pth'))
            torch.save(self.actors_target[i].state_dict(), os.path.join(save_dir, 'actor_target_' + str(i)
                                                                        + '_' + str(episode) + '.pth'))
            torch.save(self.critics[i].state_dict(),
                       os.path.join(save_dir, 'critic_' + str(episode) + '.pth'))  # 保存critic网络的参数
            torch.save(self.critics_target[i].state_dict(),
                       os.path.join(save_dir, 'critic_target_' + str(episode) + '.pth'))  # 保存目标critic网络的参数

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

    def train(self, net, sigma, rho_used):
        self.sigma = sigma
        # rho_per_epoch = 2 / (sigma ** 2)
        # self.rho_used = self.el * rho_per_epoch
        # self.rho_used = sigma2rho(sigma, self.bs, L)
        self.rho_used = rho_used
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
    agent.load(episode=935)
    mp.set_start_method('spawn', force=True)

    # Initialize server
    net = Net()
    server = Server(net=net)

    # Split Datasets
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
    # batch_size = 200
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

    # Start training ##################################################################################################################
    episode = 0
    accs1 = []
    accs2 = []
    accs3 = []
    accs4 = []
    accs5 = []
    rhos1 = []
    rhos2 = []
    rhos3 = []
    rhos4 = []
    rhos5 = []
    sigmass1 = []
    sigmass2 = []
    sigmass3 = []
    sigmass4 = []
    sigmass5 = []
    while episode < episodes:
        t1 = time.time()
        print('Episode', episode)
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

        # Tests
        if episode % 1 == 0:
            # DRL test
            print('DRL test==========================================')
            net_init = Net()
            net = Net()
            net.load_state_dict(net_init.state_dict())
            server.reset(net=net)

            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])
            global_accs_DRL = []
            acc, loss = test(net=server.net, testloader=testloader, device=server.device)
            global_accs_DRL.append(acc)

            losses = [[] for _ in range(N)]  # 每个客户每轮的损失
            for client in clients:
                client.test(server.net)  # 每个客户端使用服务器的网络进行测试
            for i, client in enumerate(clients):
                acc, avg_loss = client.outs.get()  # 获取每个客户端的测试结果
                losses[i].append(avg_loss)

            sigmas = [[] for _ in range(N)]
            rho_useds = [[] for _ in range(N)]
            rhos = [[] for _ in range(N)]

            flag = -1

            isDone = 0.0
            for CR in range(CR_Total):
                if CR == CR_Total - 1:
                    isDone = 1.0
                avgloss = 0.0
                for i, client in enumerate(clients):

                    avgloss = sum(losses[i])/len(losses[i])
                    # if CR > 1:
                    #     avgloss = 0.5 * losses[i][CR] + 0.3 * losses[i][CR - 1] + 0.2 * losses[i][CR - 2]
                    # elif CR == 0:
                    #     avgloss = losses[i][CR]
                    # elif CR == 1:
                    #     avgloss = 0.5 * losses[i][CR] + 0.5 * losses[i][CR - 1]
                    state = [CR, CR_Total - CR, client.rho_total - client.rho, client.rho, avgloss]
                    action = agent.take_action(state=state, i=i)
                    rho_used = action[0] * (rho_used_max - rho_used_min) + rho_used_min
                    # rho_used = action[0] * client.rho
                    # rho_used = np.clip(rho_used,rho_used_min,rho_used_max)
                    # if client.rho <= 1e-8:
                    #     flag = i
                    #     break
                    # if CR == 19 and client.rho > 0.0:
                    #     rho_used = np.clip(client.rho, 0.0, rho_used_max)

                    sigma = rho2sigma(rho_used, client.bs, L)
                    sigmas[i].append(sigma)
                    rho_useds[i].append(rho_used)
                    rhos[i].append(client.rho)
                    sigma = sigmas[i][CR]
                    client.sigma = sigma
                    client.train(net=server.net, sigma=client.sigma, rho_used=rho_used)
                    if CR == CR_Total - 1:
                        print(client.rho)
                # if flag != -1:
                #     for i, client in enumerate(clients):
                #         print(client.rho)
                #     CR_Total = CR+1
                #     break
                net_list = []
                for client in clients:
                    cnet = client.outs.get()
                    net_list.append(cnet)
                server.aggregate(net_list=net_list)
                acc, _ = test(net=server.net, testloader=testloader, device=server.device)
                global_accs_DRL.append(acc)

                for client in clients:  # 遍历每个客户端
                    client.test(server.net)  # 使用聚合后的模型进行测试
                for i, client in enumerate(clients):  # 遍历每个客户端
                    acc, avg_loss = client.outs.get()  # 获取客户端的测试结果
                    losses[i].append(avg_loss)

                # if flag != -1:
                #     for i, client in enumerate(clients):
                #         print(client.rho)
                #     CR_Total = CR+1
                #     break

            rho_used_sums = [sum(rho_useds[i]) for i in range(N)]

            # sigmas0 = []
            # sigmas1 = []
            # sigmas2 = []
            # rho0 = []
            # rho1 = []
            # rho2 = []
            # client0 = None
            # client1 = None
            # client2 = None
            # if clients[0].rho_total <= clients[1].rho_total and clients[0].rho_total <= clients[8].rho_total:
            #     client0 = clients[0]
            #     sigmas0 = sigmas[0]
            #     rho0 = rho_useds[0]
            #     if clients[1].rho_total <= clients[8].rho_total:
            #         client1 = clients[1]
            #         sigmas1 = sigmas[1]
            #         rho1 = rho_useds[1]
            #         client2 = clients[8]
            #         sigmas2 = sigmas[8]
            #         rho2 = rho_useds[8]
            #     else:
            #         client1 = clients[8]
            #         sigmas1 = sigmas[8]
            #         rho1 = rho_useds[8]
            #         client2 = clients[1]
            #         sigmas2 = sigmas[1]
            #         rho2 = rho_useds[1]
            # elif clients[1].rho_total <= clients[0].rho_total and clients[1].rho_total <= clients[8].rho_total:
            #     client0 = clients[1]
            #     sigmas0 = sigmas[1]
            #     rho0 = rho_useds[1]
            #     if clients[0].rho_total <= clients[8].rho_total:
            #         client1 = clients[0]
            #         sigmas1 = sigmas[0]
            #         rho1 = rho_useds[0]
            #         client2 = clients[8]
            #         sigmas2 = sigmas[8]
            #         rho2 = rho_useds[8]
            #     else:
            #         client1 = clients[8]
            #         sigmas1 = sigmas[8]
            #         rho1 = rho_useds[8]
            #         client2 = clients[0]
            #         sigmas2 = sigmas[0]
            #         rho2 = rho_useds[0]
            # else:
            #     client0 = clients[8]
            #     sigmas0 = sigmas[8]
            #     rho0 = rho_useds[8]
            #     if clients[0].rho_total <= clients[1].rho_total:
            #         client1 = clients[0]
            #         sigmas1 = sigmas[0]
            #         rho1 = rho_useds[0]
            #         client2 = clients[1]
            #         sigmas2 = sigmas[1]
            #         rho2 = rho_useds[1]
            #     else:
            #         client1 = clients[1]
            #         sigmas1 = sigmas[1]
            #         rho1 = rho_useds[1]
            #         client2 = clients[0]
            #         sigmas2 = sigmas[0]
            #         rho2 = rho_useds[0]
            # plt.clf()
            # plt.plot(list(range(CR_Total)), sigmas0, linestyle='solid')
            # plt.plot(list(range(CR_Total)), sigmas1, linestyle='dotted')
            # plt.plot(list(range(CR_Total)), sigmas2, linestyle='dashed')
            # plt.xticks([0, 5, 10, 15, 20], size=12)
            # plt.yticks(size=12)
            # plt.xlabel('Communication Round $t$', fontsize=16)
            # plt.ylabel('Noise Level $\sigma_n^t$', fontsize=16)
            # plt.legend(['Client with ' + r'$\rho_k=$' + "%.5f" % client0.rho_total,
            #             'Client with ' + r'$\rho_k=$' + "%.5f" % client1.rho_total,
            #             'Client with ' + r'$\rho_k=$' + "%.5f" % client2.rho_total], fontsize=14)
            # plt.grid()
            if not os.path.exists('./exp_data6/'):
                os.makedirs('./exp_data6/')
            # plt.savefig('./exp_data2/sigmas_DRL_test_MNIST.png')

            # plt.clf()
            # plt.plot(list(range(CR_Total)), rho0, linestyle='solid')
            # plt.plot(list(range(CR_Total)), rho1, linestyle='dotted')
            # plt.plot(list(range(CR_Total)), rho2, linestyle='dashed')
            # plt.xticks([0, 5, 10, 15, 20], size=12)
            # plt.yticks(size=12)
            # plt.xlabel('Communication Round $t$', fontsize=16)
            # plt.ylabel('Rho Level $\ rho_n^t$', fontsize=16)
            # plt.legend(['Client with ' + r'$\rho_k=$' + "%.5f" % client0.rho_total,
            #             'Client with ' + r'$\rho_k=$' + "%.5f" % client1.rho_total,
            #             'Client with ' + r'$\rho_k=$' + "%.5f" % client2.rho_total], fontsize=14)
            # plt.grid()
            # plt.savefig('./exp_data2/rho_DRL_test_MNIST.png')

            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
            rhos1.append(rhos)
            accs1.append(global_accs_DRL)
            sigmass1.append(sigmas)
            # 对比实验
            # Uniform test ==============================================================================================
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
                    client.train(net=server.net, sigma=sigma, rho_used=client.rho_total / CR_Total)
                net_list = []
                for client in clients:
                    cnet = client.outs.get()
                    net_list.append(cnet)
                server.aggregate(net_list=net_list)
                acc, _ = test(net=server.net, testloader=testloader, device=server.device)
                global_accs_Uniform.append(acc)
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
            rhos2.append(rhos)
            accs2.append(global_accs_Uniform)
            sigmass2.append(sigmas)

            # Exp test ==============================================================================================
            net = Net()
            net.load_state_dict(net_init.state_dict())
            server.reset(net=net)
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])

            sigmas = [[] for _ in range(N)]
            rho_useds = [[] for _ in range(N)]
            decay_ratio = 2 ** (1 / (CR_Total - 1))

            isDone = 0.0
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
                    rho_cur = sigma2rho(sigma=sigma, bs=client.bs, L=L)
                    client.train(net=server.net, sigma=sigma, rho_used=rho_cur)
                net_list = []
                for client in clients:
                    cnet = client.outs.get()
                    net_list.append(cnet)
                server.aggregate(net_list=net_list)
                acc, _ = test(net=server.net, testloader=testloader, device=server.device)
                global_accs_Exp.append(acc)
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
            rhos4.append(rhos)
            accs4.append(global_accs_Exp)
            sigmass4.append(sigmas)

            # Linear test ==============================================================================================
            net = Net()
            net.load_state_dict(net_init.state_dict())
            server.reset(net=net)
            for i, client in enumerate(clients):
                client.reset(rho_total=rhos_init[i])

            sigmas = [[] for _ in range(N)]
            rho_useds = [[] for _ in range(N)]
            decays = [2 * client.rho_total / (CR_Total * (CR_Total - 1) * 3) for client in clients]

            isDone = 0.0
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
                    rho_cur = sigma2rho(sigma, client.bs, L)
                    client.train(net=server.net, sigma=sigma, rho_used=rho_cur)
                net_list = []
                for client in clients:
                    cnet = client.outs.get()
                    net_list.append(cnet)
                server.aggregate(net_list=net_list)
                acc, _ = test(net=server.net, testloader=testloader, device=server.device)
                global_accs_Linear.append(acc)
            for i, client in enumerate(clients):
                rhos[i].append(client.rho)
            rhos5.append(rhos)
            accs5.append(global_accs_Linear)
            sigmass5.append(sigmas)

        # 保存实验图
        avg_accs1 = [np.mean([l[CR] for l in accs1]) for CR in range(CR_Total + 1)]
        avg_accs2 = [np.mean([l[CR] for l in accs2]) for CR in range(CR_Total + 1)]
        # avg_accs3 = [np.mean([l[CR] for l in accs3]) for CR in range(CR_Total)]
        avg_accs4 = [np.mean([l[CR] for l in accs4]) for CR in range(CR_Total + 1)]
        avg_accs5 = [np.mean([l[CR] for l in accs5]) for CR in range(CR_Total + 1)]
        print('acc DRL:', avg_accs1[-1])
        print('acc Uniform:', avg_accs2[-1])
        # print('acc random:', avg_accs3[-1])
        print('acc Exp:', avg_accs4[-1])
        print('acc Arithmetic:', avg_accs5[-1])
        # plt.clf()
        # plt.plot(list(range(CR_Total + 1)), avg_accs1, linestyle='solid')
        # plt.plot(list(range(CR_Total + 1)), avg_accs2, linestyle='dotted')
        # # plt.plot(list(range(CR_Total)), avg_accs3)
        # plt.plot(list(range(CR_Total + 1)), avg_accs4, linestyle='dashed')
        # plt.plot(list(range(CR_Total + 1)), avg_accs5, linestyle='dashdot')

        # plt.xticks([0, 5, 10, 15, 20], size=12)
        # plt.yticks(size=12)
        # plt.xlabel('Communication Round $t$', fontsize=16)
        # plt.ylabel('Test Accuracy', fontsize=16)
        # plt.legend(['Our method', 'Uniform', 'Exponential', 'Arithmetic sequence'], fontsize=14)
        # plt.grid()
        # plt.savefig('./exp_data2/Acc_Test_MNIST.png')
        print('Time spent for this episode', time.time() - t1)

        episode += 1

    # 保存实验数据
    exp_data = f"accs1:{accs1}, accs2:{accs2}, accs4:{accs4}, accs5:{accs5},\
                rhos1:{rhos1}, rhos2:{rhos2}, rhos4:{rhos4}, rhos5:{rhos5},\
                sigmass1:{sigmass1}, sigmass2:{sigmass2}, sigmass4:{sigmass4}, sigmass5:{sigmass5}"
    exp_data = json.dumps(exp_data)
    with open("./exp_data6/exp_data_MNIST.json", "a") as fp:  # Pickling
        fp.write("\n")
        json.dump(exp_data, fp)
        fp.write("\n")

        # Terminate all clients
    for client in clients:
        client.terminate_process()
    for client in clients:
        client.process.join()

    print('Time spend', time.time() - t0)
