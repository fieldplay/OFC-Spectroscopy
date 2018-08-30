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
        self.time = np.linspace(-self.timeAMP, self.timeAMP, self.timeDIM)[:, np.newaxis]

        self.field_t = 5e-2*np.exp(-(self.time**2)/(2*(70.**2)))*np.cos(1.03*self.time) + 0j
        self.field_t = np.ascontiguousarray(self.field_t)
        self.gamma = np.ascontiguousarray(self.gamma)
        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(self.rho_0)
        self.energies = np.ascontiguousarray(self.energies)
        self.H0 = np.diag(self.energies)
        self.rho = np.ascontiguousarray(np.zeros_like(self.rho_0, dtype=np.complex))
        self.Lfunc = np.zeros_like(self.rho, dtype=np.complex)
        self.dyn_rho = np.ascontiguousarray(np.zeros((len(energies), self.timeDIM), dtype=np.complex))
        self.dyn_coh = np.ascontiguousarray(np.zeros((len(energies), self.timeDIM), dtype=np.complex))
        self.pol2 = np.ascontiguousarray(np.zeros_like(self.timeDIM, dtype=np.complex))

    def propagate(self):
        Propagate(
            self.rho, self.dyn_rho, self.dyn_coh, self.field_t, self.gamma, self.mu, self.rho_0, self.energies,
            self.timeDIM, self.dt
        )
        return self.rho


if __name__ == '__main__':

    energies = np.array((0., 1., 2.))
    rho_0 = np.zeros((len(energies), len(energies)), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j
    mu = np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)
    gamma = np.asarray([[0.0, 0.2, 0.1], [0.2, 0.0, 0.15], [0.1, 0.15, 0.0]]) * 1e-4
    np.fill_diagonal(gamma, 0.)

    ThreeLevel = dict(
        energies=energies,
        gamma=gamma,
        mu=mu,
        rho_0=rho_0,
        timeDIM=500000,
        timeAMP=500.
    )

    molecule = RhoPropagate(**ThreeLevel)
    molecule.propagate()

    np.set_printoptions(precision=4)
    # for i, m in enumerate(molecule.field_t):
    #     print i, molecule.field_t[i]

    print(molecule.rho)
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes[0].plot(molecule.time, molecule.field_t, 'r')

    axes[1].plot(molecule.time, molecule.dyn_rho[0].real, 'r')
    axes[1].plot(molecule.time, molecule.dyn_rho[1].real, 'b')
    axes[1].plot(molecule.time, molecule.dyn_rho[2].real, 'k')

    axes[2].plot(molecule.time, molecule.dyn_coh[0].real, 'r')
    axes[2].plot(molecule.time, molecule.dyn_coh[1].real, 'b')
    axes[2].plot(molecule.time, molecule.dyn_coh[2].real, 'k')
    plt.show()

