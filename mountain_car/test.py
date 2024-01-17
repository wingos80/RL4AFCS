import pickle
import matplotlib.pyplot as plt
# open the figure.pickle file and times.pickle file
with open('figure.pickle', 'rb') as handle:
    fig = pickle.load(handle)
with open('times.pickle', 'rb') as handle:
    times = pickle.load(handle)

for key,value in times.items():
    print(key)
# print(times)
plt.show()