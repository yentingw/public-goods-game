""" state is modeled as the opponents played against """
import numpy as np
import pandas as pd
from Agent import Agent

class QAgent(Agent):

  """QAgent Class"""
  def __init__(self, cfg, id, rounds):
    Agent.__init__(self, cfg, id, rounds)
    self.qTable = np.random.rand(self.rounds, 2) * 0.01

  def choose_action(self, round, episode):
    if np.random.uniform(0, 1) < (episode):
      [q_c, q_d] = self.qTable[round]
      self.action = int(q_d > q_c)
      return self.action

    self.action = np.random.randint(2)
    return self.action

  def update_reward(self, round, reward):
    self.acc += reward
    if round == self.rounds -1:
      self.qTable[round][self.action] = (self.qTable[round][self.action] * (1 - self.alpha) + self.alpha * reward)
    else:
      self.qTable[round][self.action] = (self.qTable[round][self.action] * (1 - self.alpha) + self.alpha * (reward + self.gamma * max(self.qTable[round+1])))

