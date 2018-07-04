import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType
from Pol2Lindblad_wrapper import Propagate


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

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        self.dt = self.timeAMP * 2. / self.timeDIM
        n = np.linspace(-self.N_comb, self.N_comb - 1, 2 * self.N_comb)[np.newaxis, :]
        self.time = np.linspace(-self.timeAMP, self.timeAMP, self.timeDIM)[:, np.newaxis]

        field_M1 = np.exp(-self.tau * np.abs(self.time) - 1j * (self.omega_M1 + self.delta_omega * n) * self.time)
        field_M2 = np.exp(-self.tau * np.abs(self.time) - 1j * (self.omega_M2 + self.delta_omega * n) * self.time)
        field_M1 = field_M1.sum(axis=1)
        field_M2 = field_M2.sum(axis=1)
        self.field_t = field_M1 + field_M2
        self.field_t /= self.field_t.max()

        self.H0 = np.diag(self.energies)
        self.D_matrix = np.zeros_like(self.rho_0, dtype=np.complex)
        self.rho = np.zeros_like(self.rho_0)
        self.Lfunc = np.zeros_like(self.rho, dtype=np.complex)
        self.L_update = np.zeros_like(self.rho, dtype=np.complex)
        self.rho_t = np.empty((3, self.timeDIM))
        self.mu_t = np.empty((self.timeDIM,))

    def propagate(self):
        Propagate(self.rho, self.field_t, self.gamma, self.mu, self.rho_0, self.energies, self.timeDIM, self.dt)
        return self.rho

if __name__ == '__main__':

    rho_0 = np.zeros((3, 3), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j
    mu = np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)
    gamma = np.random.uniform(0.1, 0.3, (3, 3))*1e-1
    np.fill_diagonal(gamma, 0.)
    energies = np.array((0., 512, 1024))

    ThreeLevel = dict(
        energies=energies,
        gamma=gamma,
        mu=mu,
        rho_0=rho_0,
        omega_M1=7,
        omega_M2=1,
        delta_omega=10,
        N_comb=10,
        tau=5e-2,
        timeDIM=500000,
        timeAMP=150.
    )

    molecule = RhoPropagate(**ThreeLevel)
    molecule.propagate()

    print molecule.rho
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(molecule.time, molecule.field_t)
    plt.show()

