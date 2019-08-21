from itertools import permutations, product, combinations_with_replacement
from collections import namedtuple
from ctypes import Structure, c_double, c_int, POINTER, Array
import pickle
from eval_pol3_wrapper import pol3_total
import pickle
import numpy as np

############################################################################################
#                                                                                          #
#   Declare new types: ADict to access dictionary elements with a (.) rather than ['']     #
#                                                                                          #
############################################################################################


class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)

CTransition = namedtuple("CTransition", ["g", "mu"])

############################################################################################
#                                                                                          #
#           Declare uniform and non-uniform (localized around comb frequencies)            #
#                                                                                          #
############################################################################################


def uniform_frequency_range(params, offset=0.):
    """
    Generate uniformly spaced frequency range
    :type params: object
    :param params: 
    :return: 
    """
    return np.linspace(
        offset - params.freq_halfwidth,
        offset + params.freq_halfwidth - 2.*params.freq_halfwidth/params.N_frequency,
        params.N_frequency
    )


def nonuniform_frequency_range_3(params, freq_offset=None):
    """
    Generation of nonuniform frequency range taylored to the 3d order optical effects 
    :param params: 
    :param freq_offset:
    :return: 
    """
    omega_M1 = params.omega_M1
    omega_M2 = params.omega_M2
    # omega_M3 = params.omega_M3

    # If freq_offset has not been specified, generate all unique third order combinations
    if not freq_offset:
        freq_offset = np.array([
            sum(_) for _ in combinations_with_replacement(
                # [omega_M1, omega_M2, omega_M3, -omega_M1, -omega_M2, -omega_M3], 3
                [omega_M1, omega_M2, -omega_M1, -omega_M2], 3
            )
        ])

        # get number of digits to round to
        decimals = np.log10(np.abs(freq_offset[np.nonzero(freq_offset)]).min())
        decimals = int(np.ceil(abs(decimals))) + 4

        # round of
        np.round(freq_offset, decimals, out=freq_offset)

        freq_offset = np.unique(freq_offset)
        print(freq_offset)

    # def points_for_lorentzian(mean):
    #     # L = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 1.])
    #     L = np.array([0, 0.05, 0.4, 1.])
    #     L = np.append(-L[::-1], L[1::])
    #
    #     return mean + 4. * params.gamma * L

    # lorentzians_per_comb_line = np.hstack(points_for_lorentzian(_) for _ in freq_offset)
    lorentzians_per_comb_line = freq_offset

    lorentzians_per_comb_line = lorentzians_per_comb_line[:, np.newaxis]

    # Positions of all comb lines
    position_comb_lines = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]

    freq = lorentzians_per_comb_line + position_comb_lines
    freq = freq.reshape(-1)
    freq = freq[np.nonzero(
        (- params.freq_halfwidth < freq) & (freq < params.freq_halfwidth)
    )]
    freq.sort()

    return np.ascontiguousarray(freq)


def get_polarization3(molecule, params, modulations):
    """
    Return the third order polarization for a specified molecule 
    :param molecule: an instance of ADict describing molecule 
    :param params: an inparams.central_freq + stance of ADict specifying calculation parameters
    :return: numpy.array -- polarization
    """

    # introducing aliases
    transition = molecule.transitions
    energy = molecule.energies

    # initialize output array with zeros
    polarization = np.zeros(params.freq.size, dtype=np.complex)
    polarization_mnv = np.zeros_like(polarization)

    for m, n, v in permutations(range(1, len(energy)), 3):
        try:
            # calculate the product of the transition dipole if they are not zeros
            mu_product = transition[(0, v)].mu * transition[(v, n)].mu * \
                         transition[(n, m)].mu * transition[(m, 0)].mu

            # reset the polarization because C-code performs "+="
            polarization_mnv[:] = 0.
            print(modulations, m, n, v)
            pol3_total(
                polarization_mnv, params,
                modulations[0], modulations[1], modulations[2],
                energy[n] - energy[v] + 1j * transition[(n, v)].g,
                energy[m] - energy[v] + 1j * transition[(m, v)].g,
                energy[v] - energy[0] + 1j * transition[(v, 0)].g,
                energy[n] - energy[0] + 1j * transition[(n, 0)].g,
                energy[m] - energy[0] + 1j * transition[(m, 0)].g,
                energy[m] - energy[n] + 1j * transition[(m, n)].g,
                energy[n] - energy[m] + 1j * transition[(n, m)].g,
                energy[m] - energy[0] + 1j * transition[(m, 0)].g,
                energy[m] - energy[0] + 1j * transition[(m, 0)].g,

            )
            polarization_mnv *= mu_product
            polarization += polarization_mnv
        except KeyError:
            # Not allowed transition, this diagram is not calculated
            pass

    return polarization


def comb_plot(frequency, value, ax, *args, **kwargs):
    """
    Plotting with comb structure 
    :param frequency: 
    :param value: 
    :param kwargs: 
    :return: 
    """
    for omega, val in zip(frequency, value):
        ax.plot((omega, omega), (0, val), *args, **kwargs)
    ax.plot(frequency, np.zeros_like(value), 'k', linewidth=2.)
    # plt.plot(frequency, value)


def linear_spectra(molecule, omega):
    """
    Calculates linear spectra from linear susceptibility and molecular parameters
    :param molecule: 
    :return: 
    """
    transition = molecule.transitions
    energy = molecule.energies

    spectra = np.sum([transition[(n, 0)].mu**2*(energy[n] - energy[0])*transition[(n, 0)].g
                      / ((energy[n] - energy[0] - omega)**2 + transition[(n, 0)].g**2) for n in range(1, len(energy))]
                     , axis=0)
    print(spectra)
    return spectra


############################################################################################
#
#   Run test
#
############################################################################################


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    delta_freq = 1
    comb_size = 10
    transition = CTransition(0.5, 1.)

    molecule = ADict(

        #  Energy difference of levels (should be multiples of delta_freq)
        energies=np.cumsum([0, 3, 2.01e3, 2]) * delta_freq,

        # dipole value and line width for each transition
        transitions={
            (0, 1): transition,
            (1, 0): transition,
            (0, 2): transition,
            (2, 0): transition,
            (0, 3): transition,
            (3, 0): transition,
            (1, 2): transition,
            (2, 1): transition,
            (1, 3): transition,
            (3, 1): transition,
            (2, 3): transition,
            (3, 2): transition
        }
    )

    params = ADict(
        N_frequency=200,
        comb_size=comb_size,
        freq_halfwidth=2.*delta_freq*comb_size,
        omega_M1=0.07,
        omega_M2=0.03,
        gamma=5e-6,
        delta_freq=delta_freq,
        width_g=6.
    )

    import time
    start = time.time()
    frequency = nonuniform_frequency_range_3(params)
    print(frequency)
    # frequency = uniform_frequency_range(params, offset=0)
    params['freq'] = frequency

    print(params.freq.size)

    print(time.time() - start)

    omega = frequency[:, np.newaxis]
    # gaussian = np.exp(-(np.arange(-params.comb_size, params.comb_size)) ** 2 / (2.*params.width_g ** 2))[np.newaxis, :]
    comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
    # field1 = (gaussian*(params.gamma / ((omega - params.omega_M1 - comb_omega)**2 + params.gamma**2))).sum(axis=1)
    # field2 = (gaussian*(params.gamma / ((omega - params.omega_M2 - comb_omega)**2 + params.gamma**2))).sum(axis=1)
    field1 = (params.gamma / ((omega - params.omega_M1 - comb_omega) ** 2 + params.gamma ** 2)).sum(axis=1)
    field2 = (params.gamma / ((omega - params.omega_M2 - comb_omega) ** 2 + params.gamma ** 2)).sum(axis=1)

    def plot_all_modulations():
        all_modulations = list(
            product(*(3 * [[params.omega_M1, params.omega_M2]]))
        )

        pol3 = np.zeros((9, params.freq.size), dtype=np.complex)
        for i, modulations in enumerate(all_modulations):
            print(i, modulations)
            pol3[i] = get_polarization3(molecule, params, modulations)

        pol3_sum = pol3[1] + pol3[6]

        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
        fig.suptitle("$(b_1)$-term contribution to total $P^{(3)}(\\omega)$")
        for i in range(3):
            for j in range(3):
                if (i != 2) or (j != 2):
                    axes[i, j].plot(frequency / delta_freq, np.abs(pol3[3 * i + j]), 'k', linewidth=2.)
                    axes[i, j].set_ylabel(
                        'Modulations = {}, {}, {} \n'.format(*all_modulations[3 * i + j]) + '$P^{(3)}(\\omega)$',
                        color='k')
                    axes[i, j].tick_params('y', colors='k')
                    ax2 = axes[i, j].twinx()
                    ax2.plot(frequency, field1, 'b', alpha=0.6)
                    ax2.plot(frequency, field2, 'r', alpha=0.6)
                    ax2.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
                    ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
                    ax2.tick_params('y', colors='b')

        axes[2, 2].plot(frequency, pol3_sum.real, 'k', linewidth=2.)
        axes[2, 2].set_ylabel('All modulations \n' + '$P^{(3)}(\\omega)$', color='k')
        axes[2, 2].tick_params('y', colors='k')
        ax2 = axes[2, 2].twinx()
        ax2.plot(frequency, field1, 'b', alpha=0.6)
        ax2.plot(frequency, field2, 'r', alpha=0.6)
        ax2.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
        ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
        ax2.tick_params('y', colors='b')
        fig.subplots_adjust(wspace=0.30, hspace=0.00)
        print(time.time() - start)
        plt.show()

    def plot_no_modulations(k):
        pol3 = get_polarization3(molecule, params, [params.omega_M2, params.omega_M2, params.omega_M1])
        # pickle.dump({"pol3_2_20000_3": pol3}, open("Plots/pol23.p", "wb"))
        ax[k].plot(frequency / delta_freq, np.abs(pol3), 'k', linewidth=1.)
        # comb_plot(frequency / delta_freq, pol3, ax[k], 'k', linewidth=2.)
        ax[k].set_ylabel('$P^{(3)}(\\omega)$', color='k')
        ax[k].tick_params('y', colors='k')
        ax[k].set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
        ax2 = ax[k].twinx()
        comb_plot(frequency / delta_freq, field1, ax2, 'b', alpha=0.3, linewidth=0.5)
        comb_plot(frequency / delta_freq, field2, ax2, 'r', alpha=0.3, linewidth=0.5)

        pol3_matrix[:, k] = pol3
        # ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
        # ax2.tick_params('y', colors='b')

    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("$P^{(3)}(\\omega)$ with level structure 0 - 2 - 200000 - 3")

    pol3_matrix = np.empty((params.freq.size, 3), dtype=np.complex)

    for i in range(3):
        molecule.energies = np.cumsum([0, 3+2*i, 2.01e3, 2+2*i]) * delta_freq
        plot_no_modulations(i)

    print(pol3_matrix)

    with open("pol3_matrix.pickle", "wb") as f:
        pickle.dump(
            {
                'pol3_matrix': np.abs(pol3_matrix),
                'freq': frequency/delta_freq
            }, f

        )

    with open("pol3_matrix.pickle", "rb") as f:
        data = pickle.load(f)

    frequency = data['freq']
    pol3_mat = data['pol3_matrix'].real

    # frequency = np.linspace(2.01e3, 2.02e3, 10000)
    # plt.figure()
    # plt.plot(frequency, linear_spectra(molecule, frequency), 'r')
    # molecule.energies = np.cumsum([0, 3, 2.01e3, 3]) * delta_freq
    # plt.plot(frequency, linear_spectra(molecule, frequency), 'b')
    # molecule.energies = np.cumsum([0, 3, 2.01e3, 2]) * delta_freq
    # plt.plot(frequency, linear_spectra(molecule, frequency), 'g')
    plt.show()