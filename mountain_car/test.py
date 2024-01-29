import pickle
import matplotlib.pyplot as plt
import numpy as np

# open the figure.pickle file and times.pickle file
# with open('figure3.pickle', 'rb') as handle:
#     fig = pickle.load(handle)
with open('times3.pickle', 'rb') as handle:
    times = pickle.load(handle)

for key,value in times.items():
    print(key)


for key, value in times.items():
    avg = np.mean(times['1'],axis=0)
    moving_avg = 0.1*avg[3:] + 0.2*avg[2:-1] + 0.3*avg[1:-2] + 0.4*avg[:-3]
    moving_avg = moving_avg

    plt.plot(moving_avg,label=f'{key}-step SARSA')

plt.yscale('log')
plt.ylim(80,2500)
plt.ylabel('Timesteps per episode')
plt.xlabel('Episode')
plt.legend()
plt.grid()
plt.show()



# print(times)
# plt.show()