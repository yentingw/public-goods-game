import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Plotter(object):
  def __init__(self, agent_config, game_config):
    self.agent_type_list = game_config['agent_type_list']
    self.population_size = game_config['population_size']
    self.group_size = game_config['group_size']
    self.rounds = game_config['rounds']
    self.F = game_config['F']
    self.M = game_config['M']
    self.cost = game_config['cost']
    self.agents = {}
    self.dir = game_config['output_directory']
    self.logger = game_config['logger']
    self.csv = game_config['csv']
    self.plot = game_config['plot']
    self.alpha = agent_config['alpha']
    self.gamma = agent_config['gamma']
    self.epsilon = agent_config['epsilon']

    self.heatmap = pd.DataFrame(columns=['F', 'M', 'q_value'])
    self.grand_table = [[0, 0] for i in range(self.population_size)]

  # def q_plot(self, df):
  #   q_table = df[['Round', 'C%']]
  #   q_table.plot(x ='Round', y='C%', kind='line')
  #   filename = f"q_plot_test"
  #   save_path = f"{self.dir}/{filename}.png"
  #   saveplot(filename, "Number of Rounds", "Percentage of C", save_path, self.rounds)

  def QJ_plot(self, df_q, df_j):
    q_table = df_q[['Round', 'C%']]
    j_table = df_j[['Round', 'C%']]

    ax = plt.gca()
    q_table.plot(x ='Round', y='C%', kind='line', label="Q-Learning Agent", ax=ax)
    j_table.plot(x ='Round', y='C%', kind='line', label="Joint Action Q-Learning Agent", ax=ax)
    saveplot("Average Cooperation Rate Over 50 Episodes", "Number of Rounds", "C%", f"{self.dir}/Average QJ Agent.png", self.rounds)

  # def plotAverAgent(self, df, agent_type):
  #   q_table = df[['round', 'q_value']]
  #   aver_table = pd.DataFrame(columns=['round', 'q_value'])

  #   for i in range(self.rounds):
  #     aver_table = aver_table.append({'round': i, 'q_value': 0}, ignore_index=True)

  #   for i in range(self.population_size):
  #     for j in range(self.rounds):
  #       aver_table.iloc[j, 1] += q_table.iloc[i*self.rounds+j, 1]

  #   for i in range(self.rounds):
  #     aver_table.iloc[i, 1] /= self.population_size

  #   c = aver_table.iloc[self.rounds-1, 1]
  #   ls = [self.F, self.M ,c, self.group_size, self.population_size, self.alpha, self.gamma, self.epsilon]
  #   heatmap_path = f"{self.dir}/heatmap_{agent_type}.csv"
  #   with open(heatmap_path, 'a', newline='') as f:
  #     writer = csv.writer(f)
  #     writer.writerow(ls)
  #     f.close()

  #   filename = f"Average {agent_type}"
  #   save_path = f"{self.dir}/{filename}.png"
  #   aver_table.plot(x ='round', y='q_value', kind='line', label=agent_type)
  #   saveplot(filename, "Number of Rounds", "Percentage of C", save_path, self.rounds)

  # def plotAverQJ(self, df_q, df_j):
  #   q_table = df_q[['round', 'q_value']]
  #   j_table = df_j[['round', 'q_value']]

  #   aver_q_table = pd.DataFrame(columns=['round', 'q_value'])
  #   aver_j_table = pd.DataFrame(columns=['round', 'q_value'])
  #   for i in range(self.rounds):
  #     aver_q_table = aver_q_table.append({'round': i, 'q_value': 0}, ignore_index=True)
  #     aver_j_table = aver_j_table.append({'round': i, 'q_value': 0}, ignore_index=True)

  #   for i in range(self.population_size):
  #     for j in range(self.rounds):
  #       aver_q_table.iloc[j, 1] += q_table.iloc[i*self.rounds+j, 1]
  #       aver_j_table.iloc[j, 1] += j_table.iloc[i*self.rounds+j, 1]

  #   for i in range(self.rounds):
  #     aver_q_table.iloc[i, 1] /= self.population_size
  #     aver_j_table.iloc[i, 1] /= self.population_size

  #   ax = plt.gca()
  #   aver_q_table.plot(x ='round', y='q_value', kind='line', label="Q-Learning Agent", ax=ax)
  #   aver_j_table.plot(x ='round', y='q_value', kind='line', label="Joint Action Q-Learning Agent", ax=ax)
  #   saveplot("Average Performance", "Number of Rounds", "Percentage of C", f"{self.dir}/Average QJ Agent.png", self.rounds)


def saveplot(title, xlabel, ylabel, figname, rounds):
  plt.legend()
  plt.xticks(np.arange(0, rounds+1, 10.0))
  plt.yticks(np.arange(0, 1.1, 0.1))
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.savefig(figname)

