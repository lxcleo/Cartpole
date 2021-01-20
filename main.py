import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
#Self-built functions
import DQN
from Experience_replay import ReplayMemory
import setup as st


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: 
	from IPython import display

plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

# Initialize parameters
batch_size = 256
# Discount factor used in bellman equation
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001
# Update target network every 10 epsides
target_update = 10
# Number of frames being stored in the replay memory
memory_size = 10000
# Learning rate
lr = 0.001
num_epsiodes = 1000


env = st.EnvManager(device)
init_screen = env.get_screen()








 