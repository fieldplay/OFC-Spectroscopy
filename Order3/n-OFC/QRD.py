import numpy as np
import matplotlib.pyplot as plt
import pickle

from CalculatePolarization3 import comb_plot

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

N_comb = 10
rows, cols = pol3_mat.shape
print(rows, cols)

sigma = 0.16
x = np.linspace(0., 1., rows)
x = np.linspace(0., 1., 2 * N_comb)

colors = ['r', 'b', 'k']

det = np.empty((3, 3))
det_noise = np.empty((3, 3))

het_field = np.empty([cols, 2 * N_comb])


def pol2_basis_change_freq2comb():
    """
    CALCULATES BASIS CHANGE MATRIX FROM FREQUENCY BASIS TO COMB BASIS: DIM - (N_frequency x N_comb)
    :return: POLARIZATION MATRIX IN THE COMB BASIS: DIM - (N_frequency x N_molecules)
    """
    omega_cb = frequency[:, np.newaxis]
    del_omega1_cb = 1 * np.arange(- N_comb, N_comb)
    comb_omega1_cb = del_omega1_cb[np.newaxis, :]

    freq2comb_basis = 5e-6 / (
            (omega_cb - 0.07 - 0.03 - comb_omega1_cb) ** 2 + 5e-6 ** 2
    )
    freq2comb_basis /= freq2comb_basis.max()
    pol3_comb_matrix = freq2comb_basis.T.dot(pol3_mat)
    return pol3_comb_matrix


for i in range(3):
    Q_mat, R_mat = np.linalg.qr(np.delete(pol2_basis_change_freq2comb(), i, 1), mode='complete')

    print(Q_mat.real)
    gaussian = np.sin(2 * (1+i) * np.pi * (x - x.min()) / x.max()) * np.exp(-(x - 0.5) ** 2 / (2 * sigma ** 2))
    het_field[i] = sum(q * np.vdot(q, gaussian) for q in Q_mat[:, 3:].T)
    het_field[i] = sum(q for q in Q_mat[:, 3:].T)

fig, axes = plt.subplots(nrows=2, ncols=3)

for i in range(3):
    comb_plot(frequency, pol3_mat[:, i] / pol3_mat.max(), axes[0, i], colors[i], alpha=0.75, linewidth=2.)
    comb_plot(np.arange(20), het_field[i], axes[1, i], colors[i], alpha=0.75, linewidth=2.)

    axes[0, i].set_ylim(-0.025, 1.05)
    axes[1, i].set_ylim(-3.525, 1.625)
plt.show()
