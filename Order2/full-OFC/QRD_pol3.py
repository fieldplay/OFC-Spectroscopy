import numpy as np
import matplotlib.pyplot as plt
import pickle

from CalculatePolarization2_full import comb_plot

with open("pol3_matrix.pickle", "rb") as f:
    data = pickle.load(f)

frequency = data['freq']
pol3_mat = data['pol3_matrix'].real


def plot_pol3():

    fig, axes = plt.subplots(nrows=2, ncols=1)
    comb_plot(frequency, pol3_mat[:, 0].real, axes[0], 'k')
    comb_plot(frequency, pol3_mat[:, 1].real, axes[0], 'b')
    comb_plot(frequency, pol3_mat[:, 2].real, axes[0], 'r')
    axes[0].plot(frequency, np.zeros_like(frequency), 'k')

    comb_plot(frequency, pol3_mat[:, 0].imag, axes[1], 'k')
    comb_plot(frequency, pol3_mat[:, 1].imag, axes[1], 'b')
    comb_plot(frequency, pol3_mat[:, 2].imag, axes[1], 'r')
    axes[1].plot(frequency, np.zeros_like(frequency), 'k')

# plot_pol3()

rows, cols = pol3_mat.shape
print(rows, cols)
sigma = 0.15
x = np.linspace(0., 1., rows)

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
colors = ['r', 'b', 'k']

det = np.empty((3, 3))
det_noise = np.empty((3, 3))

for i in range(3):
    Q_mat, R_mat = np.linalg.qr(np.delete(pol3_mat, i, 1), mode='complete')

    # gaussian = (1. / (sigma * np.sqrt(np.pi))) * np.exp(-(x - 0.5) ** 2 / (2 * sigma ** 2))
    #   gaussian *= np.random.rand(gaussian.size)

    gaussian = np.sin(4 * np.pi * (x - x.min()) / x.max()) * np.exp(-(x - 0.5) ** 2 / (2 * sigma ** 2))
    # gaussian = np.exp(-(x - 0.5) ** 2 / (2 * sigma ** 2))
    het_field = sum(q * np.vdot(q, gaussian) for q in Q_mat[:, 3:].T)
    det[i] = np.abs(np.asarray([np.vdot(pol3_mat[:, j], het_field) for j in range(3)]))
    det[i] /= det[i][i]
    het_field *= np.random.uniform(0.95, 1.05, het_field.shape)
    het_field_1 = het_field.copy()
    het_field_1[::2] = 0.0
    het_field_2 = het_field.copy()
    het_field_2[1::2] = 0.0

    comb_plot(frequency[::2], pol3_mat[:, i][::2], axes[0], colors[i], alpha=0.5, linewidth=1.)
    comb_plot(frequency[1::2], pol3_mat[:, i][1::2], axes[1], colors[i], alpha=0.5, linewidth=1.)
    axes[0].plot(frequency[::2], pol3_mat[:, i][::2], color=colors[i], linewidth=2.)
    axes[1].plot(frequency[1::2], pol3_mat[:, i][1::2], color=colors[i], linewidth=2.)

    comb_plot(frequency, het_field_1, axes[2], colors[i], alpha=0.5, linewidth=1.)
    comb_plot(frequency, het_field_2, axes[3], colors[i], alpha=0.5, linewidth=1.)
    axes[2].plot(frequency[1::2], het_field.copy()[1::2], color=colors[i], linewidth=2.)
    axes[3].plot(frequency[::2], het_field.copy()[::2], color=colors[i], linewidth=2.)

    det_noise[i] = np.abs(np.asarray([np.vdot(pol3_mat[:, j], het_field) for j in range(3)]))
    det_noise[i] /= det_noise[i][i]

print(det)
print(det_noise)
print(1./np.linalg.det(det))
print(1./np.linalg.det(det_noise))

plt.show()
