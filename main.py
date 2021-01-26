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
memory_size= 10000
# Learning rate
lr = 0.001
num_epsiodes = 1000


env = st.EnvManager(device)
init_screen = env.get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.num_action()
memory = ReplayMemory(memory_size)
episode_duration = []
policy_net = DQN.DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN.DQN(screen_height, screen_width, n_actions).to(device)
# Map each layers with its parameter tesnor using stat_dict
# load_state_dict will load parameters of a NN
target_net.load_state_dict(policy_net.state_dict())
# eval() will let torch know it is not in training mode
target_net.eval()
#Define optimization method
optimizer = optim.SGD(params = policy_net.parameters(), lr = lr)
def optimize_model():
	if len(memory) < batch_size:
		return 
	transitions = memory.sample(batch_size)
	batch = Transition(*zip(*transitions))
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),\
		device = device, dtype = torch.unit8)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	state_action_values = policy_net(state_batch).gather(1,action_batch)
	next_state_values = torch.zeros(batch_size, device = device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	expected_state_action_values = (next_state_values * gamma) + reward_batch

	loss = F.smooth_l1_loss(state)
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp(-1,1)

	optimizer.step()


strategy = st.Epsilon(eps_start, eps_end, eps_decay)
agent = st.ActionSelection(strategy, n_actions, device)

nun_epsiodes = 20
for i in range(num_epsiodes):
	env.reset()
	last_screen = env.get_screen()
	current_screen = env.get_screen()
	state = current_screen- last_screen
	for t in count():
		action = agent.selection(state, policy_net)
		reward = env.take_action(action)
		last_screen = current_screen
		current_screen = env.get_screen()

		if not env.done:
			next_state = current_screen - last_screen

		else:
			next_state = None


		memory.push(state,action, next_state, reward)

		state = next_state

		optimize_model()
		if env.done:
			episode_duration.append(t+1)
			st.plot(episode_duration, 100)
			break


	if i % target_update == 0:
		target_net.load_state_dict(policy_net.state_dict())



env.close()















 