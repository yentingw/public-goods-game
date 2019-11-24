import numpy as np
import pandas as pd
import math
from Agent import Agent

class JointQAgent(Agent):
  """docstring for ClassName"""

  def __init__(self, cfg, id, rounds):
    Agent.__init__(self, cfg, id, rounds)
    self.qTable = np.random.rand(self.rounds, 2, 16) * 0.01
    self.fitness = 0
    self.q_info = pd.DataFrame(columns=['round', 'agent_type', 'agent_id', 'acc_reward', 'q_value'])

  def choose_action(self, round, history, opp_list, episode, history_id):
    pop_index = 0
    for i in range(len(history_id)):
      if history_id[i] == self.id:
        pop_index = i

    #if np.random.uniform(0, 1) > (1 * self.epsilon ** round):
    if np.random.uniform(0, 1) < (episode):
      [q_c, q_d] = [jointQ(qList, history, opp_list, pop_index, round+1) for qList in self.qTable[round]]

      self.action = int(q_d > q_c)
      return self.action

    self.action = np.random.randint(2)
    return self.action

  def update_reward(self, round, reward, actions):
    self.fitness += reward
    opp_actions = []
    for (i, a) in enumerate(actions):
      if a[0] != self.id:
        opp_actions.append(a[1])
    index = toDecimal(opp_actions)
    if round == self.rounds - 1:
      self.qTable[round][self.action][index] = (self.qTable[round][self.action][index] * (1 - self.alpha) + self.alpha * reward)
    else:
      self.qTable[round][self.action][index] = (self.qTable[round][self.action][index] * (1 - self.alpha) + self.alpha * (reward + self.gamma * maxQ(self.qTable[round+1])))

def jointQ(qList, history, opp_list, pop_index, round):
  temp = history[pop_index]
  del(history[pop_index])

  acc = 0
  ls = []
  for i in range(len(qList)):
    opp_actions = [int(b) for b in bin(i)[2:].zfill(4)]
    p1 = history[0][opp_actions[0]] / round
    p2 = history[1][opp_actions[1]] / round
    p3 = history[2][opp_actions[2]] / round
    p4 = history[3][opp_actions[3]] / round
    p = p1 * p2 * p3 * p4
    if p == 0:
      p = np.random.uniform(0, 1)
    #acc += qList[i] * p1 * p2 * p3 * p4
    v = qList[i] * p
    ls.append(v)
  history.insert(pop_index, temp)
  #return acc
  return max(ls)

def toDecimal(actions):
  binary = [str(a) for a in actions]
  return int("".join(binary),2)

def maxQ(ls):
  m = -math.inf
  for l in ls:
    m = max(max(l), m)
  return m

