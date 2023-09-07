import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import torch.autograd as autograd
import random

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class QNetwork(nn.Module):
    def __init__(self, name, env, input_shape, num_actions):
        super(QNetwork, self).__init__()
        self.env = env
        self.input_shape = input_shape
        self.num_actions = num_actions
        robust = False
        self.robust = robust
        width = 1
        if name == 'DQN':
            self.features = nn.Sequential(
                nn.Linear(input_shape[0], 128 * width),
                nn.ReLU(),
                nn.Linear(128 * width, 128 * width),
                nn.ReLU(),
                nn.Linear(128 * width, self.env.action_space.n)
            )
        elif name == 'CnnDQN':
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 32 * width, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32 * width, 64 * width, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64 * width, 64 * width, kernel_size=3, stride=1),
                nn.ReLU(),
                Flatten(),
                nn.Linear(3136 * width, 512 * width),
                nn.ReLU(),
                nn.Linear(512 * width, self.num_actions)
            )
        elif name == 'DuelingCnnDQN':
            self.features = DuelingCnnDQN(input_shape, num_actions, width)
        else:
            raise NotImplementedError('{} network structure not implemented.'.format(name))

    def forward(self, *args, **kwargs):
        return self.features(*args, **kwargs)

    def act(self, state, epsilon=0):
        q_value = self.forward(state)
        action = q_value.max(1)[1].data.cpu().numpy()
        mask = np.random.choice(np.arange(0, 2), p=[1 - epsilon, epsilon])
        action = (1 - mask) * action + mask * np.random.randint(self.env.action_space.n, size=state.size()[0])

        return action


class DuelingCnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, width=1):
        super(DuelingCnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape, 32 * width, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32 * width, 64 * width, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64 * width, 64 * width, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136 * width, 512 * width),
            nn.ReLU(),
            nn.Linear(512 * width, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(3136 * width, 512 * width),
            nn.ReLU(),
            nn.Linear(512 * width, 1)
        )

    def forward(self, x):
        cnn = self.cnn(x)
        advantage = self.advantage(cnn)
        value = self.value(cnn)
        return value + advantage - torch.sum(advantage, dim=1, keepdim=True) / self.num_actions

    def act(self, state, epsilon=0):
        q_value = self.forward(state)
        action = q_value.max(1)[1].data.cpu().numpy()
        mask = np.random.choice(np.arange(0, 2), p=[1 - epsilon, epsilon])
        action = (1 - mask) * action + mask * np.random.randint(self.num_actions, size=state.size()[0])

        return action


def model_setup(env):
    net_name = 'DuelingCnnDQN'
    model = QNetwork(net_name, env, env.observation_space.shape[0], env.action_space.shape[0])
    model = model.cuda()

    return model


# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


# DQN
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, input_shape, out_shape):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 288)
        self.fc2 = nn.Linear(288, 128)
        self.fc3 = nn.Linear(128, out_shape)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        # x = x.clone().detach().to(device=device)
        print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


STACK_SIZE = 5


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.linear = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        return self.head(x)
