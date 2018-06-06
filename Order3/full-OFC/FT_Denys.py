import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from pylab import gca

__doc__ = """
Compare different ways of computing the continuous Fourier transform
"""

print(__doc__)

############################################################################
#
#   parameters defining the coordinate grid
#
############################################################################

X_gridDIM = 1024*2**4
X_amplitude = 6.

############################################################################

k = np.arange(X_gridDIM)
dX = 2 * X_amplitude / X_gridDIM

# the coordinate grid
X = (k - X_gridDIM / 2) * dX

############################################################################
#
#   plot the original
#
############################################################################

# randomly generate the width of the gaussian
alpha = 1.25
# randomly generate the displacement of the gaussian
a = np.random.uniform(0., 0.2 * X_amplitude)

# original function
# f = np.exp(-alpha * (X - a) ** 2)
A = np.random.uniform(-2., 2, (8,))
phi = np.random.uniform(-np.pi, np.pi, (8,))
w0 = (10, 50, 90, 130, 170, 210, 250, 290)
print w0

X0 = 3.
# f = 2. * np.exp(- 0.1 * (X/0.5) ** 2)*np.cos(30.*X + np.pi/2.)
# f = np.exp(- alpha * np.abs(X)) * np.cos(a*X + b)
f = sum([A[i] * np.exp(- alpha * np.abs(X)) * np.cos(w0[i]*X + phi[i]) for i in range(8)])


FT_exact = lambda p, a, w, b: sum(
    [A[i]*(np.exp(1j*b[i])*np.exp(-1j*(P-w[i])*X0)*alpha/(alpha**2 + (P-w[i])**2)
           + np.exp(-1j*b[i])*np.exp(-1j*(P+w[i])*X0)*alpha/(alpha**2 + (P+w[i])**2)) for i in range(8)]
)

plt.subplot(211)
plt.title('E(t)')
plt.plot(X, f, linewidth=2.)
plt.xlabel('$t$')
plt.ylabel('$\\exp(-\\alpha |x|)$')


############################################################################
#
#   correct method : Use the first method from
#   http://epubs.siam.org/doi/abs/10.1137/0915067
#
############################################################################
minus = (-1) ** k
FT_approx1 = dX * minus * fftpack.fft(minus * f, overwrite_x=True)

# get the corresponding momentum grid
P = (k - X_gridDIM / 2) * (np.pi / X_amplitude)

plt.subplot(223)
plt.title("E($\\nu$) -- REAL")
plt.plot(P, FT_approx1.real, 'k-.', label='real approximate', linewidth=2.)
plt.plot(P, FT_exact(P, A, w0, phi).real, 'r', label='real exact', linewidth=1.)
plt.xlim(-800, 800)
plt.xlabel('$\\nu$')

plt.subplot(224)
plt.title("E($\\nu$) -- IMAGINARY")
plt.plot(P, FT_approx1.imag, 'k-.', label='imag approximate', linewidth=2.)
plt.plot(P, FT_exact(P, A, w0, phi).imag, 'r', label='real exact', linewidth=1.)
plt.xlim(-400, 400)
plt.xlabel('$\\nu$')

plt.show()
