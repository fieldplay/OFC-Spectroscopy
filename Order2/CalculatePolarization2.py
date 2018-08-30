from itertools import permutations, product, combinations_with_replacement
from collections import namedtuple
from ctypes import Structure, c_double, c_int, POINTER, Array

from eval_pol2_wrapper import pol2_total

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


def uniform_frequency_range(params):
    """
    Generate uniformly spaced frequency range
    :type params: object
    :param params: 
    :return: 
    """
    return np.linspace(
        - params.freq_halfwidth, params.freq_halfwidth - 2.*params.freq_halfwidth/params.N_frequency,
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
    polarization_mn = np.zeros_like(polarization)

    for m, n in permutations(range(1, len(energy))):
        try:
            # calculate the product of the transition dipole if they are not zeros
            mu_product = transition[(0, n)].mu * transition[(n, m)].mu * transition[(m, 0)].mu

            # reset the polarization because C-code performs "+="
            polarization_mn[:] = 0.
            print(modulations, m, n)
            pol2_total(
                polarization_mn, params,
                modulations[0], modulations[1],
                energy[n] - energy[0] + 1j * transition[(n, 0)].g,
                energy[m] - energy[0] + 1j * transition[(m, 0)].g,
                energy[m] - energy[n] + 1j * transition[(m, n)].g,
                energy[n] - energy[m] + 1j * transition[(n, m)].g,

            )
            polarization_mn *= mu_product
            polarization += polarization_mn
        except KeyError:
            # Not allowed transition, this diagram is not calculated
            pass

    return polarization


def comb_plot(frequency, value, *args, **kwargs):
    """
    Plotting with comb structure 
    :param frequency: 
    :param value: 
    :param kwargs: 
    :return: 
    """
    # for omega, val in zip(frequency, value):
    #     plt.plot((omega, omega), (0, val), *args, **kwargs)
    plt.plot(frequency, value)

############################################################################################
#
#   Run test
#
############################################################################################


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    # Energy difference of levels
    E_10 = 2.354e6
    E_21 = 2.354e6

    molecule = ADict(

        energies=[0., E_10, E_21 + E_10],

        # dipole value and line width for each transition
        transitions={
            (0, 1): CTransition(4.5e3, 1.),
            (1, 0): CTransition(4.5e3, 1.),
            (0, 2): CTransition(3.5e3, 1.),
            (2, 0): CTransition(3.5e3, 1.),
            (1, 2): CTransition(2.5e3, 1.),
            (2, 1): CTransition(2.5e3, 1.),
        }
    )

    params = ADict(
        N_frequency=2000,
        comb_size=10,
        freq_halfwidth=2.5e6,
        omega_M1=0.5,
        omega_M2=1.5,
        gamma=5,
        delta_freq=2.e5,
        width_g=5.
    )

    import time
    start = time.time()
    # frequency = nonuniform_frequency_range_3(params)
    frequency = uniform_frequency_range(params)
    params['freq'] = frequency
    print(params.freq.size)

    print(time.time() - start)

    omega = frequency[:, np.newaxis]
    gaussian = np.exp(-(np.arange(-params.comb_size, params.comb_size)) ** 2 / (2.*params.width_g ** 2))[np.newaxis, :]
    comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
    field1 = (gaussian*(params.gamma / ((omega - params.omega_M1 - comb_omega)**2 + params.gamma**2))).sum(axis=1)
    field2 = (gaussian*(params.gamma / ((omega - params.omega_M2 - comb_omega)**2 + params.gamma**2))).sum(axis=1)

    all_modulations = list(
        product(*(2 * [[params.omega_M1, params.omega_M2]]))
    )

    pol3 = np.zeros((4, params.freq.size), dtype=np.complex)
    for i, modulations in enumerate(all_modulations):
        print(i, modulations)
        pol3[i] = get_polarization3(molecule, params, modulations).real

    pol3_sum = pol3.sum(axis=0)
    # pol3_sum = pol3[1] + pol3[6]

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.suptitle("$All terms' contribution to total $P^{(2)}(\\omega)$")
    for i in range(2):
        for j in range(2):
            axes[i, j].plot(frequency, pol3[2*i+j].real, 'k', linewidth=2.)
            axes[i, j].set_ylabel('Modulations = {}, {} \n'.format(*all_modulations[2*i+j]) + '$P^{(2)}(\\omega)$', color='k')
            axes[i, j].tick_params('y', colors='k')
            ax2 = axes[i, j].twinx()
            ax2.plot(frequency, field1, 'b', alpha=0.6)
            ax2.plot(frequency, field2, 'r', alpha=0.6)
            ax2.set_xlabel("$\\omega_1 + \\omega_2 + \\Delta \\omega$ (in GHz)")
            ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
            ax2.tick_params('y', colors='b')

    fig.subplots_adjust(wspace=0.30, hspace=0.00)
    print(time.time() - start)
    plt.show()