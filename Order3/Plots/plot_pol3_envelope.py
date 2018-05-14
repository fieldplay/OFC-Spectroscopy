import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def find_envelope(file):
    with open(file + ".p") as f:
        pol = pickle.load(f)[file[:3] + "3_" + file[3] + "_20000_" + file[4]]
        pol = pol[pol.real > 1e8]
    pol = pol[np.where([abs(pol[i] - pol[i - 1]) > 100 for i in range(1, len(pol))])]
    freq = range(0, len(pol))
    print freq
    func = interp1d(freq, pol, kind='cubic')
    freq_new = np.arange(0, len(pol)-1, 0.1)
    pol_new = func(freq_new)  # use interpolation function returned by `interp1d`
    return freq_new, pol_new.real

freq33, pol33 = find_envelope("pol33")
freq32, pol32 = find_envelope("pol32")
freq23, pol23 = find_envelope("pol23")
plt.figure()
plt.plot(freq33, pol33, 'r', label='33')
plt.plot(freq32, pol32, 'b', label='32')
plt.plot(freq23, pol23, 'k', label='23')
plt.legend()
plt.show()
