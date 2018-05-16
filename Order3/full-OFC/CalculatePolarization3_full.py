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


def nonuniform_frequency_range_3(molecule, params):
    """
    Generation of nonuniform frequency range taylored to the 3d order optical effects 
    :param params: 
    :return: 
    """
    energy = molecule.energies
    omega_M1 = params.omega_M1
    omega_M2 = params.omega_M2

    # w0 = np.ceil((energy[len(energy)-1] - energy[0])/params.delta_freq)
    # w0_pol3 = np.ceil((energy[len(energy)-2] - energy[len(energy)-3])/params.delta_freq) - (2 * omega_M2 - omega_M1)
    w0_pol3 = 0. - (2 * omega_M2 - omega_M1)
    w0_field1 = 0. - omega_M1
    w0_field2 = 0. - omega_M2

    pol3_lines = []
    [pol3_lines.append(w0_pol3 + i*params.delta_freq) for i in range(params.comb_size)]
    [pol3_lines.append(w0_pol3 - i*params.delta_freq) for i in range(params.comb_size)]

    freq_pol3 = np.asarray(pol3_lines)[:, np.newaxis]\
        + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, 11)
    freq_pol3 = np.unique(np.sort(freq_pol3.reshape(-1)))

    field_lines_1 = []
    [field_lines_1.append(w0_field1 + i * params.delta_freq) for i in range(params.comb_size)]
    [field_lines_1.append(w0_field1 - i * params.delta_freq) for i in range(params.comb_size)]
    freq_field1 = np.asarray(field_lines_1)[:, np.newaxis] \
        + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, 11)
    freq_field1 = np.unique(np.sort(freq_field1.reshape(-1)))

    field_lines_2 = []
    [field_lines_2.append(w0_field2 + i * params.delta_freq) for i in range(params.comb_size)]
    [field_lines_2.append(w0_field2 - i * params.delta_freq) for i in range(params.comb_size)]
    freq_field2 = np.asarray(field_lines_2)[:, np.newaxis] \
        + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, 11)
    freq_field2 = np.unique(np.sort(freq_field2.reshape(-1)))

    return np.ascontiguousarray(freq_pol3), np.ascontiguousarray(freq_field1), np.ascontiguousarray(freq_field2)


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
    # for omega, val in zip(frequency, value):
    #     ax.plot((omega, omega), (0, val), *args, **kwargs)
    # ax.plot(frequency, np.zeros_like(value), 'k', linewidth=2.)
    ax.plot(frequency, value, *args, **kwargs)


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
    transition = CTransition(1e0, 1.)

    molecule = ADict(

        #  Energy difference of levels (should be multiples of delta_freq)
        energies=np.cumsum([0, 4, 2.01e3, 3]) * delta_freq,

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
        N_frequency=10000,
        comb_size=50,
        omega_M1=0.01,
        omega_M2=0.08,
        gamma=5e-6,
        delta_freq=delta_freq,
        width_g=6.,
        N_terms=5
    )

    import time
    start = time.time()
    frequency = nonuniform_frequency_range_3(molecule, params)[0]
    print frequency
    params['freq'] = frequency

    print params.freq.size

    print time.time() - start

    field_frequency = params.delta_freq*np.linspace(-params.comb_size, params.comb_size, params.N_frequency)
    print field_frequency
    field_frequency1 = nonuniform_frequency_range_3(molecule, params)[1]
    field_frequency2 = nonuniform_frequency_range_3(molecule, params)[2]
    omega1 = field_frequency1[:, np.newaxis]
    omega2 = field_frequency2[:, np.newaxis]
    comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
    field1 = (params.gamma / ((omega1 - params.omega_M1 - comb_omega) ** 2 + params.gamma ** 2)).sum(axis=1)
    field2 = (params.gamma / ((omega2 - params.omega_M2 - comb_omega) ** 2 + params.gamma ** 2)).sum(axis=1)

    # plt.figure()
    # plt.plot(field_frequency1/delta_freq, field1, 'r')
    # plt.plot(field_frequency2/delta_freq, field2, 'b')
    # plt.show()

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
                    comb_plot(field_frequency1 / delta_freq, field1, ax2, 'b', alpha=0.6)
                    comb_plot(field_frequency2 / delta_freq, field2, ax2, 'r', alpha=0.6)
                    ax2.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
                    ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
                    ax2.tick_params('y', colors='b')

        comb_plot(frequency / delta_freq, pol3_sum.real, axes[2, 2], 'k', linewidth=2.)
        axes[2, 2].set_ylabel('All modulations \n' + '$P^{(3)}(\\omega)$', color='k')
        axes[2, 2].tick_params('y', colors='k')
        ax2 = axes[2, 2].twinx()
        comb_plot(field_frequency1 / delta_freq, field1, ax2, 'b', alpha=0.6)
        comb_plot(field_frequency2 / delta_freq, field2, ax2, 'r', alpha=0.6)
        ax2.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
        ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
        ax2.tick_params('y', colors='b')
        fig.subplots_adjust(wspace=0.30, hspace=0.00)
        print time.time() - start

    def plot_no_modulations(ax1, clr):
        pol3 = get_polarization3(molecule, params, [params.omega_M2, params.omega_M2, params.omega_M1])
        ax1.plot(frequency / delta_freq, np.abs(pol3), color=clr, linewidth=2.)
        ax1.set_ylabel('$P^{(3)}(\\omega)$', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
        # ax2 = ax1.twinx()
        # plt.plot(field_frequency1 / delta_freq, field1, 'b', alpha=0.4)
        # plt.plot(field_frequency2 / delta_freq, field2, 'r', alpha=0.4)
        # ax2.set_ylabel('Fields $E(\\omega)$ in $fs^{-1}$', color='b')
        # ax2.tick_params('y', colors='b')
        print time.time() - start
    # plot_all_modulations()

    def plot_spacing_dependence():
        fig, axes = plt.subplots(2, 3)
        list_delta_freq = np.linspace(0.1, 0.35, 6).tolist()
        mol_energies = [np.cumsum([0, 3, 2.01e3, 3]) * delta_freq, np.cumsum([0, 3, 2.01e3, 4]) * delta_freq, np.cumsum([0, 4, 2.01e3, 3]) * delta_freq]
        colors = ['r', 'b', 'g']
        for i, delta in enumerate(list_delta_freq):
            axes[i/3, i % 3].set_title('$\Delta \omega = $' + str(list_delta_freq[i]))
            # params.delta_freq = delta
            # params['freq'] = nonuniform_frequency_range_3(molecule, params)[0]
            for j, energy in enumerate(mol_energies):
                molecule.energies = energy
                plot_no_modulations(axes[i/3, i % 3], colors[j])

    plot_spacing_dependence()

    plt.show()
