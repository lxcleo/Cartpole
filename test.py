import setup as st
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import gym
import numpy as np
import Experience_replay as ep


env = gym.make('CartPole-v0').unwrapped
env.reset()
screen = env.render('rgb_array').transpose((2,0,1))
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = st.EnvManager(device)
em.reset()
num = em.get_height()
print(num)
print(num)
print(num)
print(num)
print(num)
print(num)
print(num)
print(num)
print(num)
print(num)
st.plot(np.random.rand(300), 100)
screen = em.render('rgb_array')
screen = em.get_processed_screen()
plt.figure()
plt.imshow(screen.squeeze(0).permute(1, 2, 0), interpolation = 'none')
plt.show()
'''