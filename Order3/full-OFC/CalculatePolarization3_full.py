from itertools import permutations, product, combinations_with_replacement
from collections import namedtuple
from ctypes import Structure, c_double, c_int, POINTER, Array
import pickle
from eval_pol3_wrapper_full import pol3_total

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


def nonuniform_frequency_range_3(molecule, params):
    """
    Generation of nonuniform frequency range taylored to the 3d order optical effects 
    :param params: 
    :return: 
    """
    energy = molecule.energies

    w0 = np.ceil((energy[len(energy)-1] - energy[0])/params.delta_freq)
    w0 = 0.
    comb_lines = []
    [comb_lines.append(w0 + i*params.delta_freq) for i in range(params.comb_size)]
    [comb_lines.append(w0 - i*params.delta_freq) for i in range(params.comb_size)]

    freq = np.asarray(comb_lines)[:, np.newaxis]    # + np.linspace(-0.3*params.delta_freq, 0.3*params.delta_freq, 7)
    freq = np.unique(np.sort(freq.reshape(-1)))
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
            print modulations, m, n, v
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
    # ax.plot(frequency, np.zeros_like(value), 'k', linewidth=2.)
    # ax.plot(frequency, value, *args, **kwargs)


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
    print spectra
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

    delta_freq = 0.1
    transition = CTransition(0.5e0, 1.)

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
        N_frequency=2000,
        freq_halfwidth=1.5,
        comb_size=25,
        omega_M1=0.05,
        omega_M2=0.15,
        gamma=5e-6,
        delta_freq=delta_freq,
        width_g=6.,
        N_terms=5
    )

    import time
    start = time.time()
    frequency = nonuniform_frequency_range_3(molecule, params)
    # frequency = uniform_frequency_range(params, offset=0)
    print frequency
    params['freq'] = frequency

    print params.freq.size

    print time.time() - start

    omega = frequency[:, np.newaxis]
    comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
    field1 = (params.gamma / ((omega - params.omega_M1 - comb_omega) ** 2 + params.gamma ** 2)).sum(axis=1)
    field2 = (params.gamma / ((omega - params.omega_M2 - comb_omega) ** 2 + params.gamma ** 2)).sum(axis=1)

    def plot_all_modulations():
        all_modulations = list(
            product(*(3 * [[params.omega_M1, params.omega_M2]]))
        )

        pol3 = np.zeros((9, params.freq.size), dtype=np.complex)
        for i, modulations in enumerate(all_modulations):
            print i, modulations
            pol3[i] = get_polarization3(molecule, params, modulations)

        pol3_sum = pol3[1] + pol3[6]

        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
        fig.suptitle("$Total $P^{(3)}(\\omega)$")
        for i in range(3):
            for j in range(3):
                if (i != 2) or (j != 2):
                    comb_plot(frequency / delta_freq, np.abs(pol3)[3 * i + j], axes[i, j], 'k', linewidth=2.)
                    axes[i, j].set_ylabel(
                        'Modulations = {}, {}, {} \n'.format(*all_modulations[3 * i + j]) + '$P^{(3)}(\\omega)$',
                        color='k')
                    axes[i, j].tick_params('y', colors='k')
                    ax2 = axes[i, j].twinx()
                    comb_plot(frequency / delta_freq, field1, ax2, 'b', alpha=0.6)
                    comb_plot(frequency / delta_freq, field2, ax2, 'r', alpha=0.6)
                    ax2.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
                    ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
                    ax2.tick_params('y', colors='b')

        comb_plot(frequency / delta_freq, pol3_sum.real, axes[2, 2], 'k', linewidth=2.)
        axes[2, 2].set_ylabel('All modulations \n' + '$P^{(3)}(\\omega)$', color='k')
        axes[2, 2].tick_params('y', colors='k')
        ax2 = axes[2, 2].twinx()
        comb_plot(frequency / delta_freq, field1, ax2, 'b', alpha=0.6)
        comb_plot(frequency / delta_freq, field2, ax2, 'r', alpha=0.6)
        ax2.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
        ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
        ax2.tick_params('y', colors='b')
        fig.subplots_adjust(wspace=0.30, hspace=0.00)
        print time.time() - start

    def plot_no_modulations():
        pol3 = get_polarization3(molecule, params, [params.omega_M2, params.omega_M2, params.omega_M1])
        fig, ax1 = plt.subplots()
        comb_plot(frequency / delta_freq, np.abs(pol3), ax1, 'k', linewidth=2.)
        ax1.set_ylabel('$P^{(3)}(\\omega)$', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
        ax2 = ax1.twinx()
        comb_plot(frequency / delta_freq, field1, ax2, 'b', alpha=0.4)
        comb_plot(frequency / delta_freq, field2, ax2, 'r', alpha=0.4)
        ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
        ax2.tick_params('y', colors='b')
        print time.time() - start

    plot_no_modulations()
    # plot_all_modulations()

    plt.show()
