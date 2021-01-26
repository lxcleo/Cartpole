import random
from collections import namedtuple
experience = namedtuple('Experience',('state','action','next_state','reward'))
class ReplayMemory(object):
	# Capacity = N how many frames could be stored
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.count = 0

	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		else:
			self.memory[self.count] = experience(*args)
			self.count = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)



