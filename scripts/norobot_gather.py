import tensorflow as tf
tf.enable_eager_execution()
import gym
import torch
import numpy as np
import time
import real_robots
from real_robots.policy import BasePolicy
import cv2
import matplotlib.pyplot as plt 
np.random.seed(7)


class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        self.action_space = action_space
        self.action = np.zeros(action_space.shape[0])
        self.action += -np.pi*0.5

    def step(self, observation, reward, done):
        self.action += 0.4*np.pi*np.random.randn(self.action_space.shape[0])
        return self.action
env = gym.make("REALRobot-v0")
pi = RandomPolicy(env.action_space)
#env.render("human")


class Explorer():
  def __init__(self):
    self.mu = 0
    self.std = 0.2  
    
  def reset(self):
    self.mu = 0
    
  def sample(self, dim):
    smpl = np.random.randn(dim)*self.std + self.mu
    self.mu = smpl*0.95
    return smpl
    
expl = Explorer()

states, actions, next_states = [], [], []
empty_states = []
joints = []
def unroll_state(state):
    return cv2.resize(state["retina"], (64,64), interpolation = cv2.INTER_NEAREST)/255.

state = unroll_state(env.reset())
start = time.time()
for t in range(30000):
  #  if t % 1000 == 0: 
      #  print("time", t, "took", time.time() - start)
       # start = time.time()
    action = expl.sample(9)
    action[0] = np.clip(action[0], -1.,1.)
    action[1] = np.clip(action[1], 1.,2.)
    state, reward, done, info = env.step(action)
    #joints.append(state["joint_positions"])
    if t%100 == 0:
        state = unroll_state(state)
        states.append(state)
        for j in range(20):
            action = np.zeros(9)
            state, reward, done, info = env.step(action)
        empty_states.append(unroll_state(state))
    #actions.append(action)
    if t % 300 == 0: 
        print("time", t, "took", time.time() - start)
        start = time.time()
        expl.reset()
    if t % 500 == 0: 
      state = unroll_state(env.reset())
    if t % 3000 == 0:
      np.save("states_couple.npy", np.array(states))
      np.save("empty_states_couple.npy", np.array(empty_states))
