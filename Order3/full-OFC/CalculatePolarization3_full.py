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
    w0_pol3_21 = 0. + (2 * omega_M2 - omega_M1)
    w0_pol3_12 = 0. + (2 * omega_M1 - omega_M2)
    w0_field1 = 0. - omega_M1
    w0_field2 = 0. - omega_M2

    N = params.comb_size
    pol3_lines = []
    [pol3_lines.append(w0_pol3_21 + i*params.delta_freq) for i in range(params.comb_size)]
    [pol3_lines.append(w0_pol3_21 - i*params.delta_freq) for i in range(params.comb_size)]
    [pol3_lines.append(w0_pol3_12 + i * params.delta_freq) for i in range(params.comb_size)]
    [pol3_lines.append(w0_pol3_12 - i * params.delta_freq) for i in range(params.comb_size)]

    # pol3_lines = np.hstack((w0_pol3_12 + np.linspace(-N, N, 2*N + 1), w0_pol3_21 + np.linspace(-N, N, 2*N + 1)))
    freq_pol3 = np.asarray(pol3_lines)[:, np.newaxis]\
        # + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, 21)
    freq_pol3 = np.sort(np.unique(freq_pol3.reshape(-1)))

    field_lines_1 = []
    [field_lines_1.append(w0_field1 + i * params.delta_freq) for i in range(params.comb_size)]
    [field_lines_1.append(w0_field1 - i * params.delta_freq) for i in range(params.comb_size)]
    freq_field1 = np.asarray(field_lines_1)[:, np.newaxis] \
        + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, 11)
    freq_field1 = np.sort(np.unique(freq_field1.reshape(-1)))

    field_lines_2 = []
    [field_lines_2.append(w0_field2 + i * params.delta_freq) for i in range(params.comb_size)]
    [field_lines_2.append(w0_field2 - i * params.delta_freq) for i in range(params.comb_size)]
    freq_field2 = np.asarray(field_lines_2)[:, np.newaxis] \
        + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, 11)
    freq_field2 = np.sort(np.unique(freq_field2.reshape(-1)))

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
                energy[v] - energy[n] + 1j * transition[(m, 0)].g,
                energy[v] - energy[m] + 1j * transition[(m, 0)].g,

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

    delta_freq = .1
    transition = CTransition(1e1, 1.)

    molecule = ADict(

        #  Energy difference of levels (should be multiples of delta_freq)
        energies=np.cumsum([0, 4, 2.01e3, 3]) * delta_freq,

        # dipole value and line width for each transition
        transitions={
            (0, 1): CTransition(1.25e0, 1.),
            (1, 0): CTransition(1.25e0, 1.),
            (0, 2): CTransition(1.25e0, 1.),
            (2, 0): CTransition(1.25e0, 1.),
            (0, 3): CTransition(2.25e0, 1.),
            (3, 0): CTransition(2.25e0, 1.),
            (1, 2): CTransition(2.25e0, 1.),
            (2, 1): CTransition(2.25e0, 1.),
            (1, 3): CTransition(1.25e0, 1.),
            (3, 1): CTransition(1.25e0, 1.),
            (2, 3): CTransition(1.25e0, 1.),
            (3, 2): CTransition(1.25e0, 1.)
        }
    )

    params = ADict(
        N_frequency=1000,
        comb_size=250,
        omega_M1=0.05,
        omega_M2=0.07,
        gamma=5e-6,
        delta_freq=delta_freq,
        width_g=6.,
        N_terms=5
    )

    import time
    start = time.time()
    frequency = nonuniform_frequency_range_3(molecule, params)[0]
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

    def plot_all_modulations(axes, clr):
        all_modulations = list(
            product(*(3 * [[params.omega_M1, params.omega_M2]]))
        )

        pol3 = np.zeros((8, params.freq.size), dtype=np.complex)
        for i, modulations in enumerate(all_modulations):
            if (i == 1) or (i == 6):
                print i, modulations
                pol3[i] = get_polarization3(molecule, params, modulations)

        pol3 /= pol3.max()
        pol3_sum_field_free = pol3[1] + pol3[6]
        comb_plot(frequency / delta_freq, pol3_sum_field_free.real, axes, clr, linewidth=1.)
        return pol3_sum_field_free

    def plot_no_modulations(ax1, clr):
        pol3 = get_polarization3(molecule, params, [params.omega_M2, params.omega_M2, params.omega_M1])
        ax1.plot(frequency / delta_freq, np.abs(pol3), color=clr, linewidth=2.)
        ax1.set_ylabel('$P^{(3)}(\\omega)$', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
        print time.time() - start

    def plot_spacing_dependence():
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        list_delta_freq = np.linspace(0.1, 0.60, 6).tolist()
        mol_energies = [np.cumsum([0, 3, 2.01e3, 3]) * delta_freq, np.cumsum([0, 3, 2.01e3, 4]) * delta_freq, np.cumsum([0, 4, 2.01e3, 3]) * delta_freq]
        colors = ['r', 'b', 'g']
        for i, delta in enumerate(list_delta_freq):
            axes[i/3, i % 3].set_title('$\Delta \omega = $' + str(list_delta_freq[i]))
            params.delta_freq = delta
            for j, energy in enumerate(mol_energies):
                molecule.energies = energy
                plot_no_modulations(axes[i/3, i % 3], colors[j])

    # plot_spacing_dependence()

    def plot_L_spectra_NL_pol3(molecule, ax, clr):
        frequency = np.linspace(3.5e6 - 10, 3.5e6 + 15, 10000)
        spectra = linear_spectra(molecule, frequency)
        ax.plot(frequency, spectra, clr, linewidth=2.)
        # pol3 = get_polarization3(molecule, params, [params.omega_M2, params.omega_M2, params.omega_M1])
        # ax.plot(frequency, pol3.real, clr, linewidth=1., alpha=0.6)


    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    fig1, axes1 = plt.subplots(nrows=1, ncols=1)
    fig.suptitle("Total $P^{(3)}(\\omega)$")
    pol3_matrix = np.empty((params.freq.size, 3), dtype=np.complex)
    molecule.energies = np.cumsum([0, 26, 3.5e7, 35]) * delta_freq
    plot_L_spectra_NL_pol3(molecule, axes1, 'k')
    pol3_matrix[:, 0] = plot_all_modulations(axes, 'k')
    molecule.energies = np.cumsum([0, 25, 3.5e7, 36]) * delta_freq
    plot_L_spectra_NL_pol3(molecule, axes1, 'b')
    pol3_matrix[:, 1] = plot_all_modulations(axes, 'b')
    molecule.energies = np.cumsum([0, 24, 3.5e7, 37]) * delta_freq
    plot_L_spectra_NL_pol3(molecule, axes1, 'r')
    pol3_matrix[:, 2] = plot_all_modulations(axes, 'r')
    axes.set_xlabel('$(\omega - \omega_{central})/ \Delta \omega$', color='k')
    axes.set_ylabel('Field-free polarizations \n' + '$P^{(3)}(\\omega)$', color='k')
    axes.tick_params('y', colors='k')
    print time.time() - start
    plt.show()

    print frequency.max()/delta_freq, frequency.min()/delta_freq

    with open("pol3_matrix.pickle", "wb") as f:
        pickle.dump(
            {
                'pol3_matrix': pol3_matrix,
                'freq': frequency/delta_freq
            }, f

        )


