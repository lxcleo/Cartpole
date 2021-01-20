import gym
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

resize = T.compose([T.ToPILImage(), T.resize(40, interpolation=Image.CUBIC), T.ToTensor()])
class EnvManager():
	def __init__(self, device):
		self.env = gym.make('CartPole-v0').unwrapped
		self.device = device



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


	def reset(self):
		self.env.reset()








	