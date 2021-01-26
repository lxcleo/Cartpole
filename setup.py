import gym
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])
class EnvManager():
	def __init__(self, device):
		self.env = gym.make('CartPole-v0').unwrapped
		self.device = device
		self.done = False



	def get_cart_location(self, screen_width):
		world_width = self.env.x_threshold * 2
		scale = screen_width / world_width
		return int(self.env.state[0] * scale + screen_width / 2.0)


	def render(self, mode = 'human'):
		return self.env.render(mode)



	def get_screen(self):
		self.reset()
		screen = self.render('rgb_array').transpose((2, 0, 1))
		_, screen_height, screen_width = screen.shape
		# Get ride of top and bottom redundent part
		screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
		view_width = int(screen_height * 0.6)
		cart_loation = self.get_cart_location(screen_width)
		# // 向下整除
		if cart_loation < view_width // 2:
			slice_range = slice(view_width)
		elif cart_loation > (screen_width - view_width // 2):
			slice_range = slice(-view_width, None)
		else:
			slice_range = slice(cart_loation - view_width // 2, cart_loation + view_width //2)

		screen = screen[:, :, slice_range]
		screen = np.ascontiguousarray(screen, dtype = np.float32) / 255
		screen = torch.from_numpy(screen)
		return resize(screen).unsqueeze(0).to(self.device)


	def num_action(self):
		return self.env.action_space.n




	def take_action(self, action):
		_, reward, self.done, _ = self.env.step(action.item())
		return torch.tensor([reward], device = self.device)


	def reset(self):
		self.env.reset()


	def close(self):
		self.env.close()



class Epsilon():
	def __init__(self,start,end,decay):
		self.start = start
		self.end = end
		self.decay = decay

	def get_EpsiolonRate(self,step):
		rate = self.end + (self.start-self.end) * \
		math.exp(-1. * step * self.decay)
		return rate




class ActionSelection():
	def __init__(self,strategy,num_actions, device):
		self.current_step = 0
		self.strategy = strategy
		self.num_actions = num_actions
		self.device = device
	def selection(self,state,policy_net):
		rate = self.strategy.get_EpsiolonRate(self.current_step)
		self.current_step += 1

		if rate > random.random():
			action = random.randrange(self.num_actions) 
			return torch.tensor([action]).to(self.device)# explore

		else:
			# Do not track on the gradient 
			with torch.no_grad():
				return policy_net(state).max(1)[1].view(1,1).to(self.device) # exploit




def plot(values, moving_avg_period):
	plt.figure(2)
	plt.clf()
	plt.title('Training..')
	plt.xlabel('# of Episode')
	plt.ylabel('Duration')
	plt.plot(values)
	plt.plot(get_moving_average(moving_avg_period, values))
	plt.pause(0.001)
	if is_ipython: display.clear_output(wait=True)


