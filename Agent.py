""" state is modeled as the opponents played against """
import numpy as np

class Agent(object):
  """docstring for ClassName"""
  def __init__(self, cfg, id, rounds):
    self.logger = cfg['logger']
    self.id = id
    self.episodes = cfg['episodes']
    self.rounds = rounds
    self.alpha = cfg['alpha']
    self.gamma = cfg['gamma']
    self.epsilon = cfg['epsilon']
    self.acc = cfg['accumulative_reward']

  def choose_action(self, round, episode):
    pass

  def update_reward(self, round, reward):
    pass

  def getId(self):
    return self.id

  def print(self):
    pass
