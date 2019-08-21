import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType


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

        # self.omega_gridDIM = 50000
        # self.omega_amplitude = int(2 * self.delta_omega*self.N_comb)
        # self.k = np.arange(self.omega_gridDIM)
        # self.d_omega = 2. * self.omega_amplitude / self.omega_gridDIM
        # self.omega = (self.k - self.omega_gridDIM / 2) * self.d_omega
        #
        # minus = (-1) ** self.k
        # self.t = (self.k - self.omega_gridDIM / 2) * (np.pi / self.omega_amplitude)
        # self.dt = 2.*np.pi/self.d_omega
        #
        # print self.dt
        # print self.d_omega
        # field_omega1 = (self.tau / ((self.omega[:, np.newaxis] - self.omega_M1 - n * self.delta_omega) ** 2 + self.tau ** 2)).sum(axis=1)
        # field_omega2 = (self.tau / ((self.omega[:, np.newaxis] - self.omega_M2 - n * self.delta_omega) ** 2 + self.tau ** 2)).sum(axis=1)
        #
        # field_t1 = (self.d_omega * minus * fftpack.fft(minus * field_omega1, overwrite_x=True))
        # field_t2 = (self.d_omega * minus * fftpack.fft(minus * field_omega2, overwrite_x=True))
        # seluy mn  plot(self.omega, field_omega1)
        # plt.plot(self.omega, field_omega2)
        # plt.subplot(212)
        # self.field_t /= self.field_t.max()
        # plt.plot(self.t, self.field_t)

        self.timeDIM = 5000
        self.timeAMP = 100.
        self.dt = self.timeAMP * 2. / self.timeDIM
        n = np.linspace(-self.N_comb, self.N_comb-1, 2 * self.N_comb)[np.newaxis, :]

        self.time = np.linspace(-self.timeAMP, self.timeAMP-self.dt, self.timeDIM)[:, np.newaxis]
        exact_FT_1 = np.pi * np.exp(-self.tau*np.abs(self.time) - 1j*(self.omega_M1 + self.delta_omega*n)*self.time)
        exact_FT_1 = exact_FT_1.sum(axis=1)

        exact_FT_2 = np.pi * np.exp(-self.tau * np.abs(self.time) - 1j * (self.omega_M2 + self.delta_omega * n) * self.time)
        exact_FT_2 = exact_FT_2.sum(axis=1)

        exact_FT = exact_FT_1 + exact_FT_2
        exact_FT /= exact_FT.max()

        plt.figure()

        plt.subplot(311)
        plt.plot(self.time, exact_FT_1, 'r')
        plt.plot(self.time, exact_FT_2, 'b')

        self.H0 = np.diag(self.energies)
        self.H = self.H0.copy()
        self.D_matrix = np.zeros_like(self.rho_0, dtype=np.complex)
        self.rho = self.rho_0.copy()
        self.Lfunc = np.zeros_like(self.rho, dtype=np.complex)
        self.L_update = np.zeros_like(self.rho, dtype=np.complex)
        self.rho_t = np.empty((3, self.timeDIM))
        self.mu_t = np.empty((self.timeDIM,))
        self.field_t = exact_FT

    def L_operator(self, Qmat, indx):
        self.Lfunc[:] = 0.
        energy = self.energies
        gamma = self.gamma
        mu = self.mu

        for m in range(energy.size):
            for n in range(energy.size):

                self.Lfunc[m, n] += -1j*(energy[m] - energy[n]) * Qmat[m, n]
                for j in range(energy.size):
                    if m == n:
                        self.Lfunc[m, n] += gamma[j, m] * Qmat[j, j]
                    self.Lfunc[m, n] -= 0.5 * (gamma[n, j] + gamma[m, j]) * Qmat[m, n]
                    self.Lfunc[m, n] += 1j * self.field_t[indx] * (mu[m, j] * Qmat[j, n] - Qmat[m, j] * mu[j, n])

        # print "Lfunc trace" + str(np.trace(self.Lfunc))
        return self.Lfunc

    def propagate(self):

        # for i in range(self.timeDIM):
        for i in range(self.timeDIM):
            print(i)
            self.L_update = self.rho.copy()
            for j in range(1, 8):
                self.L_update += self.L_operator(self.L_update, i) * self.dt / j
            self.rho += self.L_update
            self.rho /= np.trace(self.rho)

            self.rho_t[:, i] = np.diag(self.rho).real
            self.mu_t[i] = np.trace(self.rho.dot(self.mu)).real

        print(self.rho)


if __name__ == '__main__':

    rho_0 = np.zeros((3, 3), dtype=np.complex)
    rho_0[0, 0] = 1.
    mu = np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)
    gamma = np.asarray([[0.0, 0.2, 0.1], [0.2, 0.0, 0.15], [0.1, 0.15, 0.0]])*1e-1
    energies = np.array((0., 512, 1024))

    qsys_params = dict(
        energies=energies,
        omega_M1=7,
        omega_M2=1,
        delta_omega=10,
        N_comb=10,
        rho_0=rho_0,
        mu=mu,
        gamma=gamma,
        tau=5e-2
    )

    molecule = RhoPropagate(**qsys_params)
    molecule.propagate()

    plt.subplot(312)

    plt.plot(molecule.time.sum(axis=1), molecule.rho_t[0], 'r')
    plt.plot(molecule.time, molecule.rho_t[1], 'b')
    plt.plot(molecule.time, molecule.rho_t[2], 'k')

    plt.subplot(313)
    plt.plot(molecule.time, molecule.mu_t, 'k')

    plt.show()

