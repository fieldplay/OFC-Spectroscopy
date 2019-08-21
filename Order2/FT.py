import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from pylab import gca

__doc__ = """
Compare different ways of computing the continuous Fourier transform
"""


def render_ticks(axis, labelsize):
    """
    Style plots for better representation
    :param axis: axes class of plot
    """
    plt.rc('font', weight='bold')
    axis.get_xaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelrotation=0, labelsize=labelsize)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelcolor='k', labelsize=labelsize)
    axis.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.5, b=None, which='both', axis='both')


################################################
#                                              #
#   parameters defining the coordinate grid    #
#                                              #
################################################

X_gridDIM = 65536
X_amplitude = 256

k = np.arange(X_gridDIM)
dX = 2 * X_amplitude / X_gridDIM

# the coordinate grid
X = ((k - X_gridDIM / 2) * dX)[:, np.newaxis]
N = 128
#########################
#                       #
#   plot the original   #
#                       #
#########################

alpha = 1.0 / (2.0 * (X_amplitude / 4) ** 2)
tau = 2.5e-2
w0 = np.linspace(-X_amplitude, X_amplitude, N)[np.newaxis, :]

f1 = (np.exp(-alpha * X ** 2) * tau / ((X - w0 - 0.75 * 2 * X_amplitude / N) ** 2 + tau ** 2)).sum(axis=1)
f2 = (np.exp(-alpha * X ** 2) * tau / ((X - w0 - 0.25 * 2 * X_amplitude / N) ** 2 + tau ** 2)).sum(axis=1)

#####################################################
#                                                   #
#   correct method : Use the first method from      #
#   http://epubs.siam.org/doi/abs/10.1137/0915067   #
#                                                   #
#####################################################
minus = (-1) ** k
FT_approx1 = dX * minus * fftpack.fft(minus * f1, overwrite_x=True)
FT_approx2 = dX * minus * fftpack.fft(minus * f2, overwrite_x=True)

# corresponding momentum grid
P = (k - X_gridDIM / 2) * (np.pi / X_amplitude)

FTI_approx1 = (1./dX) * minus * fftpack.ifft(minus * FT_approx1, overwrite_x=True)
FTI_approx2 = (1./dX) * minus * fftpack.ifft(minus * FT_approx2, overwrite_x=True)

fig, axes = plt.subplots(nrows=3, ncols=2)

axes[0, 0].plot(X, f1, 'r', linewidth=1.)
axes[0, 0].plot(X, f2, 'b', linewidth=1.)
axes[0, 1].plot(X, FTI_approx1.real, linewidth=1.)
axes[0, 1].plot(X, FTI_approx2.real, linewidth=1.)
axes[0, 0].set_xlabel('$\\omega$')
axes[0, 1].set_xlabel('$\\omega$')

axes[1, 0].plot(P, FT_approx1.real, 'k', label='real approximate', linewidth=1.)
axes[2, 0].plot(P, FT_approx2.real, 'r', label='real approximate', linewidth=1.)
axes[1, 1].plot(P, FT_approx1.imag, 'k', label='imag approximate', linewidth=1.)
axes[2, 1].plot(P, FT_approx2.imag, 'r', label='imag approximate', linewidth=1.)

for i in range(3):
    for j in range(2):
        render_ticks(axes[i, j], 'medium')
        if i > 0:
            axes[i, j].set_xlabel('$t$')

plt.show()
