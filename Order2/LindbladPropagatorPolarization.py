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

        self.omega_gridDIM = 10000
        self.omega_amplitude = 5.*self.delta_omega*self.N_comb
        self.k = np.arange(self.omega_gridDIM)
        self.d_omega = 2. * self.omega_amplitude / self.omega_gridDIM
        self.omega = (self.k - self.omega_gridDIM / 2) * self.d_omega

        minus = (-1) ** self.k
        self.t = (self.k - self.omega_gridDIM / 2) * (np.pi / self.omega_amplitude)
        self.dt = 2.*np.pi/self.d_omega

        print self.dt
        print self.d_omega
        # print omega.min(), omega.max()
        n = np.linspace(-self.N_comb, self.N_comb-1, 2 * self.N_comb)[np.newaxis, :]
        field_omega1 = (self.tau / ((self.omega[:, np.newaxis] - self.omega_M1 - n * self.delta_omega) ** 2 + self.tau ** 2)).sum(axis=1)
        field_omega1 *= np.blackman(field_omega1.size)
        field_omega2 = (self.tau / ((self.omega[:, np.newaxis] - self.omega_M2 - n * self.delta_omega) ** 2 + self.tau ** 2)).sum(axis=1)
        field_omega2 *= np.blackman(field_omega2.size)

        field_t1 = (self.d_omega * minus * fftpack.fft(minus * field_omega1, overwrite_x=True))
        field_t2 = (self.d_omega * minus * fftpack.fft(minus * field_omega2, overwrite_x=True))

        self.field_t = field_t1 + field_t2
        plt.figure()
        plt.subplot(311)
        plt.plot(self.omega, field_omega1, 'r-')
        plt.plot(self.omega, field_omega2, 'b-')
        plt.subplot(312)
        plt.plot(self.t, field_t1.real, 'r-')
        plt.subplot(313)
        exact_FT = np.pi * np.exp(self.tau*self.t[:, np.newaxis] - 1j*(self.omega_M1 + self.delta_omega*n))
        plt.plot(self.t, exact_FT.sum(axis=1), 'r-')

        plt.show()
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
        return self.D_matrix

    def L_operator(self, Qmat):
        return -1j*(self.H.T.dot(Qmat) - Qmat.T.dot(self.H)) + self.dissipation(Qmat)

    def propagate(self):
        self.rho_11 = np.empty((self.omega_gridDIM,))
        self.rho_22 = np.empty((self.omega_gridDIM,))
        self.rho_33 = np.empty((self.omega_gridDIM,))
        self.mu_t = np.empty((self.omega_gridDIM,))

        for i in range(self.omega_gridDIM):
            self.H = self.H0 - self.field_t[i] * self.mu
            self.Lfunc = self.rho_0.copy()
            for j in range(1, 5):
                self.Lfunc = self.L_operator(self.Lfunc)*self.dt/j
                self.rho += self.Lfunc
            # print np.trace(self.rho)

            self.rho_11[i] = self.rho[0, 0].real
            self.rho_22[i] = self.rho[1, 1].real
            self.rho_33[i] = self.rho[2, 2].real
            self.mu_t[i] = np.trace(self.rho.T.dot(self.mu)).real
        print self.rho


if __name__ == '__main__':

    rho_0 = np.zeros((3, 3), dtype=np.complex)
    rho_0[0, 0] = 1.
    mu = np.ones_like(rho_0)
    mu -= np.diag(np.ones((3,)))
    gamma = np.asarray([[0.0, 0.2, 0.1], [0.2, 0.0, 0.15], [0.1, 0.15, 0.0]])*1e-2
    energies = np.array((0., 512, 1024))

    qsys_params = dict(
        energies=energies,
        omega_M1=7,
        omega_M2=3,
        delta_omega=10,
        N_comb=100,
        rho_0=rho_0,
        mu=mu,
        gamma=gamma,
        tau=5e-1
    )

    molecule = RhoPropagate(**qsys_params)
    # molecule.propagate()
    #
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(molecule.t, molecule.rho_11, 'r')
    # plt.plot(molecule.t, molecule.rho_22, 'b')
    # plt.plot(molecule.t, molecule.rho_33, 'k')
    #
    # plt.subplot(312)
    # plt.plot(molecule.t, molecule.mu_t, 'k')


    plt.show()
