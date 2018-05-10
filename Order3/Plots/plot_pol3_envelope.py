import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("pol33.p") as f:
    pol33 = pickle.load(f)["pol3_3_20000_3"]
    pol33 = pol33[pol33.real > 0.01e10]

with open("pol32.p") as f:
    pol32 = pickle.load(f)["pol3_3_20000_2"]
    pol32 = pol32[pol32.real > 0.01e10]

with open("pol23.p") as f:
    pol23 = pickle.load(f)["pol3_2_20000_3"]
    pol23 = pol23[pol23.real > 0.01e10]

index = []
for i in range(1, len(pol33)):
    if np.abs(pol33.real[i] - pol33.real[i-1]) < 100:
        index.append(i)
index = []
for i in range(1, len(pol32)):
    if np.abs(pol32.real[i] - pol32.real[i-1]) < 100:
        index.append(i)
index = []
for i in range(1, len(pol23)):
    if np.abs(pol23.real[i] - pol23.real[i-1]) < 100:
        index.append(i)
pol33 = np.delete(pol33, index)
pol32 = np.delete(pol32, index)
pol23 = np.delete(pol23, index)


plt.figure()
plt.plot(pol33.real, 'r')

plt.plot(pol32.real, 'b')
plt.plot(pol23.real, 'k')
plt.show()
