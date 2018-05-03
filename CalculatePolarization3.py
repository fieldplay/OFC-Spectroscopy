from itertools import permutations, product, combinations_with_replacement
from collections import namedtuple
from ctypes import Structure, c_double, c_int, POINTER, Array

from eval_pol3_wrapper import pol3_total

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
        print freq_offset

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


def get_polarization3(molecule, params):
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

    # for m, n, v in permutations(range(1, len(energy))):
    for m, n, v in [(3, 1, 2)]:
        try:
            # calculate the product of the transition dipole if they are not zeros
            mu_product = transition[(0, v)].mu * transition[(v, n)].mu * \
                         transition[(n, m)].mu * transition[(m, 0)].mu

            # reset the polarization because C-code performs "+="
            polarization_mnv[:] = 0.

            # all_modulations = product(*(3 * [[params.omega_M1, params.omega_M2]]))
            all_modulations = list(
                product(*(3 * [[params.omega_M1, params.omega_M2]]))
            )
            del all_modulations[0:-1]
            # del list_mods[0]
            print all_modulations, m, n, v

            for mods in all_modulations:
                pol3_total(
                    polarization_mnv, params,
                    mods[0], mods[1], mods[2],
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
    plt.plot(frequency, value, '.')

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
    E_10 = 0.05
    E_21 = 0.05
    E_32 = 0.05

    molecule = ADict(

        energies=[0., E_10, E_21 + E_10, E_32 + E_21 + E_10],

        # dipole value and line width for each transition
        transitions={
            (0, 1): CTransition(4.5e-3, 1.),
            (1, 0): CTransition(4.5e-3, 1.),
            (0, 2): CTransition(3.5e-3, 1.),
            (2, 0): CTransition(3.5e-3, 1.),
            (0, 3): CTransition(4.0e-3, 1.),
            (3, 0): CTransition(4.0e-3, 1.),
            (1, 2): CTransition(2.5e-3, 1.),
            (2, 1): CTransition(2.5e-3, 1.),
            (1, 3): CTransition(2.0e-3, 1.),
            (3, 1): CTransition(2.0e-3, 1.),
            (2, 3): CTransition(3.0e-3, 1.),
            (3, 2): CTransition(3.0e-3, 1.)
        }
    )

    params = ADict(
        N_frequency=2000,
        comb_size=1,
        freq_halfwidth=1e2,
        omega_M1=1.,
        omega_M2=2.25,
        # omega_M3=3.,
        gamma=5e-1,
        delta_freq=2.5,
        width_g=5.
    )

    import time
    start = time.time()
    # choosing the frequency range
    # frequency = nonuniform_frequency_range_3(params)
    frequency = uniform_frequency_range(params)
    params['freq'] = frequency
    print params.freq.size

    pol3 = get_polarization3(molecule, params)
    print time.time() - start
    print pol3.imag.min(), pol3.imag.max()
    pol3 /= np.abs(pol3).max()
    print pol3.real.max()
    print pol3.imag.max()
    omega = frequency[:, np.newaxis]
    gaussian = np.exp(-(np.arange(-params.comb_size, params.comb_size)) ** 2 / (2.*params.width_g ** 2))[np.newaxis, :]
    comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
    field1 = (gaussian*(params.gamma / ((omega - params.omega_M1 - comb_omega)**2 + params.gamma**2))).sum(axis=1)
    field2 = (gaussian*(params.gamma / ((omega - params.omega_M2 - comb_omega)**2 + params.gamma**2))).sum(axis=1)
    # field3 = (gaussian*(params.gamma / ((omega - params.omega_M3 - comb_omega)**2 + params.gamma**2))).sum(axis=1)

    plt.figure()
    plt.subplot(211)
    comb_plot(frequency, field1/field1.max(), 'b-', alpha=0.5, label='Field 1')
    comb_plot(frequency, field2/field1.max(), 'r-', alpha=0.5, label='Field 2')
    # comb_plot(frequency, field3/field1.max(), 'g-', alpha=0.5, label='Field 3')
    comb_plot(frequency, 5.*pol3.real, 'k-', label='$\mathcal{R}e[P^{(3)}(\\omega)]$')

    plt.plot(frequency, np.zeros_like(frequency), 'k-')
    plt.xlabel("$\\omega_1 + \\omega_2 + \\omega_3 + \\Delta \\omega$ (in GHz)")
    plt.ylabel("$\mathcal{R}e[P^{(3)}(\\omega)]$")
    # plt.legend()

    plt.subplot(212)
    comb_plot(frequency, field1 / field1.max(), 'b-', alpha=0.5, label='Field 1')
    comb_plot(frequency, field2 / field1.max(), 'r-', alpha=0.5, label='Field 2')
    # comb_plot(frequency, field3 / field1.max(), 'g-', alpha=0.5, label='Field 3')
    comb_plot(frequency, pol3.imag, 'k-', label='$\mathcal{I}m[P^{(3)}(\\omega)]$')

    plt.plot(frequency, np.zeros_like(frequency), 'k-')
    plt.xlabel("$\\omega_1 + \\omega_2 + \\omega_3 + \\Delta \\omega$ (in GHz)")
    plt.ylabel("$\mathcal{I}m[P^{(3)}(\\omega)]$")
    # plt.legend()
    # with open("Pol3_data.pickle", "wb") as output_file:
    #     pickle.dump(
    #         {
    #             "molecules_pol3": pol3,
    #             "freq": frequency,
    #             "field1": field1,
    #             "field2": field2,
    #             "field3": field3
    #         }, output_file
    #     )

    print time.time() - start
    plt.show()