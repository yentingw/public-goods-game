import argparse
import datetime
import logging
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
import csv

from JointQAgent import JointQAgent as jointA
from QAgent import QAgent as qA
from plots import Plotter

__game_config = {}

__logger = None

class Game(object):
  def __init__(self, agent_config, game_config):
    self.agent_type_list = game_config['agent_type_list']
    self.episodes = game_config['episodes'] + 1
    self.population_size = game_config['population_size']
    self.group_size = game_config['group_size']
    self.group_number = self.population_size / self.group_size
    self.rounds = game_config['rounds']
    self.F = game_config['F']
    self.M = game_config['M']
    self.cost = game_config['cost'] / (self.F + 1)
    self.agents = {}
    self.dir = game_config['output_directory']
    self.logger = game_config['logger']
    self.plotter = game_config['plotter']
    self.timestamp = game_config['timestamp']
    self.csv = game_config['csv']
    self.plot = game_config['plot']
    self.grand_table = [[0, 0] for i in range(self.population_size)]
    self.heatmap_q_ls = []
    self.heatmap_j_ls = []
    self.plot_q_ls = [0 for i in range(self.rounds)]
    self.plot_j_ls = [0 for i in range(self.rounds)]
    self.df_q = pd.DataFrame(columns=['Round', 'C%'])
    self.df_j = pd.DataFrame(columns=['Round', 'C%'])

    for a in self.agent_type_list:
      for i in range(self.population_size):
        if a == 'QAgent':
          if 'QAgent' in self.agents:
            self.agents['QAgent'].append(qA(agent_config, i, self.rounds))
          else:
            self.agents['QAgent'] = [qA(agent_config, i, self.rounds)]
        elif a == 'JointQAgent':
          if 'JointQAgent' in self.agents:
            self.agents['JointQAgent'].append(jointA(agent_config, i, self.rounds))
          else:
            self.agents['JointQAgent'] = [jointA(agent_config, i, self.rounds)]

  def shuffle_group(self, agents):
    np.random.shuffle(agents)
    group_num = self.population_size / self.group_size
    list_arrays = np.array_split(agents, group_num)
    return [grp.tolist() for grp in list_arrays]

  def play_game(self, episodes):
    self.logger.info(f"START: {self.timestamp}")
    for e in range(1, episodes+1):
      if 'QAgent' in self.agents:
        self.q_agent_play(e, episodes)
      if 'JointQAgent' in self.agents:
        self.joint_q_agent_play(e, episodes)

    self.logger.info("Calculations finished. Begin Dumping...")
    if self.csv:
      self.logger.info("Writing to CSV")
      if 'QAgent' in self.agents:
        self.load_to_heatmap_csv(self.heatmap_q_ls, "Q_Agent")
        self.load_to_plot_csv(self.plot_q_ls, self.df_q, "Q_Agent")

      if 'JointQAgent' in self.agents:
        self.load_to_heatmap_csv(self.heatmap_j_ls, "J_Agent")
        self.load_to_plot_csv(self.plot_j_ls, self.df_j, "J_Agent")

    if self.plot:
      self.logger.info("Begin plotting")
      if 'QAgent' in self.agents and 'JointQAgent' in self.agents:
        self.plotter.QJ_plot(self.df_q, self.df_j)
      # if 'QAgent' in self.agents:
      #   self.plotter.plotAverAgent(df_q_info_q, "Q-Learning Agent")
      # if 'JointQAgent' in self.agents:
      #   self.plotter.plotAverAgent(df_q_info_j, "Joint Action Q-Learning QAgent")
    self.logger.info(f"FINISH: {datetime.datetime.utcnow().isoformat()}")

  def q_agent_play(self, episode, episodes):
    groups = self.shuffle_group(self.agents['QAgent'])
    self.logger.debug(f"Number of q groups: {len(groups)};\t agents: {[agent.id for grps in groups for agent in grps]}")
    group_num = 0
    for grp in groups:
      group_num += 1
      for r in range(self.rounds):
        actions = []
        for agent in grp:
          agent_action = agent.choose_action(r, episode / episodes)
          actions.append(agent_action)
          self.logger.debug(f"episode: {episode}, round: {r}, group_num: {group_num}, agent_id: {agent.id}, agent_type: {agent.__class__.__name__}, action: {agent_action}")

        c_count = self.count_c(actions) / self.group_size
        self.plot_q_ls[r] += c_count

        if (episode == episodes - 1) and (r == self.rounds - 1):
          self.heatmap_q_ls.append(c_count)

        rewards = self.calculate_reward(actions)
        for (agent, reward) in zip(self.agents['QAgent'], rewards):
          agent.update_reward(r, reward)


  def joint_q_agent_play(self, episode, episodes):
    j_groups = self.shuffle_group(self.agents['JointQAgent'])
    self.logger.debug(f"Number of Joint Q agent groups: {len(j_groups)};\t agents: {[agent.id for grps in j_groups for agent in grps]}")
    group_num = 0
    for j_group in j_groups:
      group_num += 1
      history_id = [agent.id for agent in j_group]
      history = []
      for i in history_id:
        history.append(self.grand_table[i])
      for r in range(self.rounds):
        actions = []
        actions_tuple = []
        for agent in j_group:
          opp_list = create_opp_list(j_group, agent)
          agent_action = agent.choose_action(r, history, opp_list, episode / episodes, history_id)
          actions.append(agent_action)
          actions_tuple.append((agent.id, agent_action))
          self.logger.debug(f"episode: {episode}, round: {r}, group_num: {group_num}, agent_id: {agent.id}, agent_type: {agent.__class__.__name__}, action: {agent_action}, opp_list: {opp_list}")

        c_count = self.count_c(actions) / self.group_size
        self.plot_j_ls[r] += c_count

        if (episode == episodes - 1) and (r == self.rounds - 1):
          self.heatmap_j_ls.append(c_count)

        if (episode % 10 == 0):

          self.df_episode_j = self.df_episode_j.append({'Episode': episode, 'Round': r, 'C%': c_count}, ignore_index=True)

        rewards = self.calculate_reward(actions)
        for i in range(len(history)):
          history[i][actions[i]] += 1
        for (agent, reward) in zip(j_group, rewards):
          agent.update_reward(r, reward, actions_tuple)

        self.logger.debug(f"history_id: {history_id}, table: {self.grand_table}")

  def count_c(self, actions):
    count = 0
    for i in actions:
      if i == 0:
        count += 1
    return count

  def load_to_heatmap_csv(self, heatmap_ls, agent_type):
    c_count = 0
    for i in heatmap_ls:
      c_count += i
    c_count /= self.group_number
    ls = [self.population_size, self.F / 5, self.M / 5, c_count]
    heatmap_path = f"outputs/heatmap_" + agent_type + ".csv"
    with open(heatmap_path, 'a', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(ls)
      f.close()

  def load_to_plot_csv(self, plot_ls, df, agent_type):
    for i in range(self.rounds):
      repeat = self.group_number * self.episodes
      plot_ls[i] /= repeat
    column1 = [i for i in range(self.rounds)]
    column2 = plot_ls
    a = np.array([column1, column2])
    a = a.T
    for i in a:
      df = df.append({'Round': i[0], 'C%': i[1]}, ignore_index=True)
    path = f"{self.dir}/plot_c%_" + agent_type + ".csv"
    df.to_csv(path)
    if agent_type == "Q_Agent":
      self.df_q = df
    if agent_type == "J_Agent":
      self.df_j = df
    #self.plotter.q_plot(df)

  def calculate_reward(self, actions):
    """ reward formula """
    num_c = 0
    for a in actions:
      if a == 0:
        num_c += 1

    sigma = 0
    delta_count = num_c - self.M

    if delta_count < 0:
      sigma = 0
    else:
      sigma = 1

    d_reward = (num_c * self.F / self.group_size * self.cost) * sigma
    c_reward = d_reward - self.cost

    return [(c_reward, d_reward)[a] for a in actions]

def create_opp_list(agents, a):
  opp_list = []
  for agent in agents:
    if agent != a:
      opp_list.append(str(agent.getId()))
  return opp_list

def toBinary(plannings):
  return [int(i) for i in bin(plannings)[2:]]

def print_agents(agents):
  df_q_info = pd.DataFrame(columns=['round', 'agent_type', 'agent_id', 'acc_reward', 'q_value'])
  df_j_info = pd.DataFrame(columns=['round', 'agent_type', 'agent_id', 'acc_reward', 'q_value'])

  for agent in agents:
    if agent.__class__.__name__ == 'QAgent':
      df_q_info = df_q_info.append(agent.get_info())
    elif agent.__class__.__name__ == 'JointQAgent':
      df_j_info = df_j_info.append(agent.get_info())


  df_q_info.to_csv(r'{}/q_info.csv'.format(self.dir), index=False)
  df_j_info.to_csv(r'{}/j_info.csv'.format(self.dir), index=False)

  #self.plotter.plotQAgent(df_q_info)
  #self.plotter.plotAverQAgent(df_q_info)


def load_ini_config():
  with open("game.yaml", 'r') as stream:
    try:
      global __game_config
      __game_config = yaml.safe_load(stream)
    except yaml.YAMLError as err:
      raise(err)


def setup_directory(cfg):
  """
  Create target Directory if it doesn't exist
  """
  directory = f"{cfg.get('output_directory', 'outputs')}-F{cfg['F']}-M{cfg['M']}-P{cfg['population_size']}"
  if not os.path.exists(directory):
    os.mkdir(directory)
  else:
    directory = f"{directory}-{cfg['timestamp']}".replace(':', '-')
    os.mkdir(directory)
  return directory


def main():
  """ main functions """
  global __logger

  load_ini_config()
  agent_config = __game_config['agents']
  game_config = __game_config['game']

  with open('logging.yaml', 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))
  __logger = logging.getLogger(__name__)
  game_config['logger'] = logging.getLogger('game')
  agent_config['logger'] = logging.getLogger('agent')

  agent_config['episodes'] = game_config['episodes']

  # Optionally Get F, M, and Population Overrides from CLI
  parser = argparse.ArgumentParser(description='F,M,Population Overrides')
  parser.add_argument('--F',
                      type=int,
                      default=game_config['F'])
  parser.add_argument('--M',
                      type=int,
                      default=game_config['M'])
  parser.add_argument('--P',
                      type=int,
                      default=game_config['population_size'])
  args = parser.parse_args()
  game_config['F'] = args.F
  game_config['M'] = args.M
  game_config['population_size'] = args.P
  __logger.info(args)

  game_config['timestamp'] = datetime.datetime.utcnow().isoformat()
  game_config['output_directory'] = setup_directory(game_config)
  game_config['plotter'] = Plotter(agent_config, game_config)

  game = Game(agent_config, game_config)
  game.play_game(game_config['episodes'])


if __name__== "__main__":
  main()
