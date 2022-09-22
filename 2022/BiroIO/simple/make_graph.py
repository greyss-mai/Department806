import pandas as pd
import matplotlib.pyplot as plt

table = 'performance_test_result.csv'

df = pd.read_csv(table, sep=';')

fig = plt.figure(figsize=(15, 10))
plt.title('Сравнение производительности CPU/GPU')
plt.plot(df['N'], df['CPU_TIME'], color='blue', label='CPU time')
plt.plot(df['N'], df['GPU_TIME'], color='red', label='GPU time')
plt.xlabel('N')
plt.ylabel('Time (microseconds)')
plt.grid()
plt.legend()
plt.savefig('performance_graph.png', dpi=300)
plt.clf()

fig = plt.figure(figsize=(15, 10))
plt.title('Кривая отношения производительност GPU/CPU')
plt.hlines(1, 0, df['N'].max(), color='red', linestyle='--', label='T = 1')
plt.plot(df['N'], df['CPU_TIME'] / df['GPU_TIME'], color='green', label='T = CPU/GPU time')
plt.xlabel('N')
plt.ylabel('T')
plt.grid()
plt.legend()
plt.savefig('performance_cruve.png', dpi=300)
plt.clf()

    