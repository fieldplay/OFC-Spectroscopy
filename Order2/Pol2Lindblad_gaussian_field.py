import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType
from Pol2Lindblad_wrapper import Propagate
from scipy import fftpack


class RhoPropagate:
    """
    Class for propagating the Lindblad Master equation. We calculate
    rho(T) and obtain NL-polarization by finding Trace[mu * rho(T)]
    """

    def __init__(self, **kwargs):
        """
        __init__ function call to initialize variables from the
        parameters for the class instance provided in __main__ and
        add new variables for use in other functions in this class.
        """

        self.fieldAMP = None
        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        self.k = np.arange(self.omegaDIM)
        self.d_omega = 2 * self.omegaAMP / self.omegaDIM
        self.omega = ((self.k - self.omegaDIM / 2) * self.d_omega)[:, np.newaxis]
        self.alpha = 1.0 / (2.0 * (self.omegaAMP / 4) ** 2)
        self.omega_0 = np.linspace(-self.omegaAMP, self.omegaAMP, self.N_comb)[np.newaxis, :]

        self.omega_M1 = 0.85 * 2 * self.omegaAMP / self.N_comb
        self.omega_M2 = 0.35 * 2 * self.omegaAMP / self.N_comb

        self.field_omega1 = (
                self.fieldAMP*np.exp(-self.alpha * self.omega ** 2)
                * self.tau / ((self.omega - self.omega_0 - self.omega_M1) ** 2 + self.tau ** 2)
                ).sum(axis=1)

        self.field_omega2 = (
                self.fieldAMP*np.exp(-self.alpha * self.omega ** 2)
                * self.tau / ((self.omega - self.omega_0 - self.omega_M2) ** 2 + self.tau ** 2)
        ).sum(axis=1)

        self.minus = (-1) ** self.k
        self.dt = np.pi / self.omegaAMP
        self.time = (self.k - self.omegaDIM / 2) * self.dt

        self.field_t1 = self.d_omega * self.minus * fftpack.fft(self.minus * self.field_omega1, overwrite_x=True)
        self.field_t2 = self.d_omega * self.minus * fftpack.fft(self.minus * self.field_omega2, overwrite_x=True)

        self.field_t = np.ascontiguousarray(self.field_t1 + self.field_t1)
        self.gamma = np.ascontiguousarray(self.gamma)
        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(self.rho_0)
        self.energies = np.ascontiguousarray(self.energies)
        self.H0 = np.diag(self.energies)
        self.rho = np.ascontiguousarray(np.zeros_like(self.rho_0, dtype=np.complex))
        self.Lfunc = np.zeros_like(self.rho, dtype=np.complex)
        self.dyn_rho = np.ascontiguousarray(np.zeros((len(energies), self.omegaDIM), dtype=np.complex))
        self.dyn_coh = np.ascontiguousarray(np.zeros((len(energies), self.omegaDIM), dtype=np.complex))
        self.pol2 = np.ascontiguousarray(np.zeros_like(self.omegaDIM, dtype=np.complex))

    def propagate(self):
        Propagate(
            self.rho, self.dyn_rho, self.dyn_coh, self.field_t, self.gamma, self.mu, self.rho_0, self.energies,
            self.omegaDIM, self.dt
        )
        return self.rho


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


if __name__ == '__main__':

    gamma = np.asarray([[0.0, 0.2, 0.1], [0.2, 0.0, 0.15], [0.1, 0.15, 0.0]])*1e-2
    np.fill_diagonal(gamma, 0.)
    energies = np.array((0., 2, 4))
    rho_0 = np.zeros((len(energies), len(energies)), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j
    mu = np.ones_like(rho_0)*2
    np.fill_diagonal(mu, 0j)

    ThreeLevel = dict(
        energies=energies,
        gamma=gamma,
        mu=mu,
        rho_0=rho_0,
        omegaAMP=321,
        omegaDIM=65536,
        N_comb=256,
        tau=.25e-1,
        fieldAMP=5e-4
    )

    molecule = RhoPropagate(**ThreeLevel)
    molecule.propagate()

    np.set_printoptions(precision=4)

    fig, axes = plt.subplots(nrows=3, ncols=2)

    response = molecule.dyn_coh.sum(axis=0).real * molecule.field_t1.real.max() / molecule.dyn_coh.sum(axis=0).real.max()
    axes[0, 0].plot(molecule.time, molecule.field_t1.real, 'r')
    axes[0, 0].plot(molecule.time, molecule.field_t2.real, 'b')
    axes[1, 0].plot(molecule.time, response, 'k', alpha=0.95)

    axes[0, 1].plot(molecule.omega, (1./molecule.d_omega) * molecule.minus * fftpack.ifft(molecule.minus * molecule.field_t1, overwrite_x=True))
    axes[0, 1].plot(molecule.omega, (1./molecule.d_omega) * molecule.minus * fftpack.ifft(molecule.minus * molecule.field_t2, overwrite_x=True))

    axes[1, 1].plot(molecule.omega, (1./molecule.d_omega) * molecule.minus * fftpack.ifft(molecule.minus * response, overwrite_x=True))
    axes[1, 1].plot(molecule.omega, (1. / molecule.d_omega) * molecule.minus * fftpack.ifft(molecule.minus * molecule.field_t1, overwrite_x=True))
    axes[1, 1].plot(molecule.omega, (1. / molecule.d_omega) * molecule.minus * fftpack.ifft(molecule.minus * molecule.field_t2, overwrite_x=True))

    axes[2, 1].plot(molecule.time, molecule.dyn_rho[0].real, 'r')
    axes[2, 1].plot(molecule.time, molecule.dyn_rho[1].real, 'b')
    axes[2, 1].plot(molecule.time, molecule.dyn_rho[2].real, 'k')

    axes[2, 0].plot(molecule.time, molecule.dyn_coh[0].real, 'r', label='$\\rho_{12}$')
    axes[2, 0].plot(molecule.time, molecule.dyn_coh[1].real, 'b', label='$\\rho_{13}$')
    axes[2, 0].plot(molecule.time, molecule.dyn_coh[2].real, 'k', label='$\\rho_{23}$')
    axes[2, 0].legend()

    for i in range(3):
        for j in range(2):
            render_ticks(axes[i, j], 'small')
    plt.show()

