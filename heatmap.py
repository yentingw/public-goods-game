import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


#df = pd.read_csv(f'outputs/P25-Q.csv', sep='\s*,\s*', header=0, encoding='ascii', engine='python')
df = pd.read_csv(f'E50R20/P500/J500.csv', sep='\s*,\s*', header=0)
q_map = pd.pivot_table(df, index=['M'], columns=['F'], values='C%')
sns.heatmap(q_map, cmap="YlGnBu")
plt.title("Q Learning, Population Size = 25")
#plt.title("Joint Action Q Learning, Population Size = 500")
plt.show()
#plt.savefig(file)
