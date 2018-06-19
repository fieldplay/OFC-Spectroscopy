import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType
from scipy import fftpack


class RhoPropagate:
    """
    
    """

    def __init__(self, **kwargs):
        """
        
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        try:
            self.energies
        except AttributeError:
            raise AttributeError("Energies not specified")

        try:
            self.gamma
        except AttributeError:
            raise AttributeError("Line-widths not specified")

        self.t_gridDIM = 2**6
        self.t_amplitude = np.pi*self.t_gridDIM / (2.*self.delta_omega*self.N_comb)
        self.k = np.arange(self.t_gridDIM)
        self.dt = 2 * self.t_amplitude / self.t_gridDIM
        self.t = ((self.k - self.t_gridDIM / 2) * self.dt)

        minus = (-1) ** self.k
        omega = (self.k - self.t_gridDIM / 2) * (np.pi / self.t_amplitude)
        # print omega.min(), omega.max()
        n = np.linspace(-self.N_comb, self.N_comb-1, 2 * self.N_comb)[np.newaxis, :]
        field_omega1 = (self.tau / ((omega[:, np.newaxis] - self.omega_M1 - n * self.delta_omega) ** 2 + self.tau ** 2)).sum(axis=1)
        field_omega2 = (self.tau / ((omega[:, np.newaxis] - self.omega_M2 - n * self.delta_omega) ** 2 + self.tau ** 2)).sum(axis=1)
        field_t1 = ((1. / self.dt) * minus * fftpack.ifft(minus * field_omega1, overwrite_x=True))
        field_t2 = ((1. / self.dt) * minus * fftpack.ifft(minus * field_omega2, overwrite_x=True))

        self.field_t = field_t1 + field_t2
        plt.figure()
        plt.subplot(211)
        plt.plot(omega, field_omega1, 'b')
        plt.plot(omega, field_omega2, 'r')
        plt.subplot(212)
        plt.plot(self.t, self.field_t, 'r')

        self.H0 = np.diag(self.energies)
        self.H = self.H0.copy()
        self.D_matrix = np.zeros_like(self.rho_0, dtype=np.complex)
        self.rho = self.rho_0.copy()
        self.Lfunc = np.zeros_like(self.rho, dtype=np.complex)

    def dissipation(self, Qmat):
        """
        Calculates effect of environment on the system
        :return: self.D_matrix: value of environment effects matrix with updated rho
        """

        self.D_matrix = np.zeros_like(self.D_matrix, dtype=np.complex)
        for i in range(self.energies.size):
            for j in range(self.energies.size):
                self.D_matrix[j, j] += Qmat[i, i]
                for k in range(self.energies.size):
                    self.D_matrix[k, i] -= 0.5*Qmat[k, i]
                    self.D_matrix[i, k] -= 0.5*Qmat[i, k]
                self.D_matrix *= self.gamma[i, j]
        # print np.trace(self.D_matrix)
        return self.D_matrix

    def L_operator(self, i, Qmat):
        self.H = self.H0 - self.field_t[i]*self.mu
        print np.trace(-1j*(self.H.dot(Qmat) - Qmat.dot(self.H)))
        return -1j*(self.H.dot(Qmat) - Qmat.dot(self.H))

    def propagate(self):
        self.Lfunc = self.rho.copy()
        for i in range(self.t_gridDIM):
            for j in range(1, 4):
                self.Lfunc = self.L_operator(i, self.Lfunc)*self.dt/j
                self.rho += self.Lfunc
            # print np.trace(self.rho)
        print self.rho


if __name__ == '__main__':

    rho_0 = np.zeros((3, 3), dtype=np.complex)
    rho_0[0, 0] = 1.
    mu = np.ones_like(rho_0)
    gamma = np.zeros_like(rho_0)
    gamma = np.random.uniform(1, 3, (3, 3))*1e-1
    energies = np.array((0., 2010, 4020))

    qsys_params = dict(
        energies=energies,
        omega_M1=13,
        omega_M2=7,
        delta_omega=16,
        N_comb=32,
        rho_0=rho_0,
        mu=mu,
        gamma=gamma,
        tau=3e-2
    )

    molecule = RhoPropagate(**qsys_params)
    molecule.propagate()
    # print molecule.dissipation(Qmat=np.random.uniform(1, 3, (3, 3))+1j*np.random.uniform(1, 3, (3, 3)))
    plt.show()
