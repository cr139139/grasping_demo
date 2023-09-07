import numpy as np
import collections
from itertools import count
from PIL import Image
import torch
import torchvision.transforms as T
from env_ros import ENVIRONMENT
import pybullet as p
from nets import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess = T.Compose([T.ToPILImage(),
                        T.Grayscale(num_output_channels=1),
                        T.Resize(40, interpolation=Image.BICUBIC),
                        T.ToTensor()])


def get_screen():
    screen = env.get_observation().transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return preprocess(screen).unsqueeze(0).to(device)


STACK_SIZE = 5
episode = 10
env = ENVIRONMENT(bc=p)
env.reset()
_, _, screen_height, screen_width = get_screen().shape

policy_net = DQN(screen_height, screen_width, env.num_actions).to(device)
checkpoint = torch.load('policy_dqn.pt')
# policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

for i_episode in range(episode):
    state = get_screen()
    stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)
    for t in count():
        stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
        action = policy_net(stacked_states_t).max(1)[1].view(1, 1)

        output = env.step(action.item())
        reward = output[1]
        done = output[2]
        next_state = get_screen()
        stacked_states.append(next_state)
        if done:
            break
    env.reset()
    print("Episode: {0:d}, reward: {1}".format(i_episode + 1, reward), end="\n")
