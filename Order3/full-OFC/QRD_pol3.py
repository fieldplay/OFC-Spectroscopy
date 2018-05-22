import numpy as np
import matplotlib.pyplot as plt
import pickle

from CalculatePolarization3_full import comb_plot

with open("pol3_matrix.pickle", "rb") as f:
    data = pickle.load(f)

frequency = data['freq']
pol3_mat = data['pol3_matrix']


def plot_pol3():
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    comb_plot(frequency, pol3_mat[:, 0].real, axes[0], 'k')
    comb_plot(frequency, pol3_mat[:, 1].real, axes[0], 'b')
    comb_plot(frequency, pol3_mat[:, 2].real, axes[0], 'r')
    axes[0].plot(data['freq'], np.zeros_like(data['freq']), 'k')
    comb_plot(frequency, pol3_mat[:, 0].imag, axes[1], 'k')
    comb_plot(frequency, pol3_mat[:, 1].imag, axes[1], 'b')
    comb_plot(frequency, pol3_mat[:, 2].imag, axes[1], 'r')
    axes[1].plot(frequency, np.zeros_like(frequency), 'k')
    axes[0].set_ylim(-2.25, 1.5)
    plt.show()

rows, cols = pol3_mat.shape
print rows, cols
sigma = 0.1
x = np.linspace(0., 1., rows)
gaussian = (1./(sigma*np.sqrt(np.pi)))*np.exp(-(x - 0.5)**2/(2*sigma**2))
# print gaussian
plt.plot(frequency, gaussian)
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

for i in range(3):
    Q_mat, R_mat = np.linalg.qr(pol3_mat[:, 1:], mode='complete')
    # for k in range(rows):
        # Q_mat[:, k] *= gaussian[k]
    het_fields = Q_mat
    print np.vdot(pol3_mat[:, i], Q_mat[:, 0])
    axes[i].plot(frequency, Q_mat[:, 3:].sum(axis=1).real)
# comb_plot(frequency, Q_mat[:, 3:].real.sum(axis=1), axes, 'k')
# comb_plot(frequency, Q_mat[:, 4].real, axes, 'b')
# comb_plot(frequency, Q_mat[:, 5].real, axes, 'r')
# comb_plot(frequency, Q_mat[:, 6].real, axes, 'g')
# comb_plot(frequency, Q_mat[:, 7].real, axes, 'y')

# for k in range(3, rows):
#     z = np.vdot(pol3_mat[:, 0], het_fields[:, k])
#     if z.real > 1e-4:
#         print z,

plt.show()
