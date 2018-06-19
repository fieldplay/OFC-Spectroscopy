from itertools import permutations, product, combinations_with_replacement
from collections import namedtuple
from ctypes import Structure, c_double, c_int, POINTER, Array
import pickle
from eval_pol2_wrapper_full import pol2_total


############################################################################################
#   Declare new types: ADict to access dictionary elements with a (.) rather than ['']     #
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
#           Declare uniform and non-uniform (localized around comb frequencies)            #
############################################################################################


def nonuniform_frequency_range_3(molecule, params):
    """
    Generation of nonuniform frequency range taylored to the 3d order optical effects 
    :param params: 
    :return: 
    """
    omega_M1 = params.omega_M1
    omega_M2 = params.omega_M2
    N = params.comb_size

    w0_pol3 = (omega_M1 + omega_M2) + params.central_freq
    w0_field1 = 2*omega_M1 + params.central_freq
    w0_field2 = 2*omega_M2 + params.central_freq

    freq_pol3 = w0_pol3 + np.linspace(-N*params.delta_freq, N*params.delta_freq, 2*N+1)
    freq_field1 = w0_field1 + np.linspace(-N*params.delta_freq, N*params.delta_freq, 2*N+1)
    freq_field2 = w0_field2 + np.linspace(-N*params.delta_freq, N*params.delta_freq, 2*N+1)

    return freq_pol3, freq_field1, freq_field2


def get_polarization2(molecule, params, modulations):
    """
    Return the second order polarization for a specified molecule 
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

    for m, n in permutations(range(1, len(energy)), 2):
        try:
            # calculate the product of the transition dipole if they are not zeros
            mu_product = transition[(0, n)].mu * transition[(n, m)].mu * transition[(m, 0)].mu

            # reset the polarization because C-code performs "+="
            polarization_mn[:] = 0.
            print modulations, m, n
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
    print transition
    print
    print energy

    spectra = np.sum([transition[(n, 0)].mu**2*(energy[n] - energy[0])*transition[(n, 0)].g
                      / ((energy[n] - energy[0] - omega)**2 + transition[(n, 0)].g**2) for n in range(1, len(energy))]
                     , axis=0)
    print spectra
    return spectra


def calculate_chi2(molecule, omega, omega1):
    assert len(omega) == len(omega1), "Frequency arrays of unequal lengths"

    N = len(omega)
    transition = molecule.transitions
    energy = molecule.energies
    chi2 = np.empty((N, N), dtype=np.complex)

    for m, n in permutations((1, 2), 2):
        print m, n
        chi2 += transition[(0, n)].mu * transition[(n, m)].mu * transition[(m, 0)].mu /\
                ((energy[n] - energy[0] - omega[:, np.newaxis] - 1j*transition[(n, 0)].g)
                 * (energy[m] - energy[0] - omega1[np.newaxis, :] - 1j*transition[(m, 0)].g))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    img = ax.imshow(chi2.real, extent=[omega.min(), omega.max(), omega1.min(), omega1.max()])
    fig.colorbar(img, ax=ax)
    plt.show()


############################################################################################
#   Run test                                                                               #
############################################################################################


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    delta_freq = 10

    molecule = ADict(

        #  Energy difference of levels (should be multiples of delta_freq)
        energies=np.cumsum([0, 2.01e5, 2.01e5]),

        # dipole value and line width for each transition
        transitions={
            (0, 1): CTransition(2.25e6, 1.),
            (1, 0): CTransition(2.75e6, 1.),
            (0, 2): CTransition(2.50e6, 1.),
            (2, 0): CTransition(2.25e6, 1.),
            (1, 2): CTransition(2.75e6, 1.),
            (2, 1): CTransition(2.25e6, 1.),
        }
    )

    params = ADict(
        central_freq=molecule.energies[2],
        comb_size=200,
        omega_M1=3,
        omega_M2=7,
        gamma=5e-4,
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

    field_frequency1 = nonuniform_frequency_range_3(molecule, params)[1]
    field_frequency2 = nonuniform_frequency_range_3(molecule, params)[2]
    omega1 = field_frequency1[:, np.newaxis]
    omega2 = field_frequency2[:, np.newaxis]
    comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
    field1 = (params.gamma / ((omega1 - params.central_freq - 2*(params.omega_M1 - comb_omega)) ** 2 + params.gamma ** 2)).sum(axis=1)
    field2 = (params.gamma / ((omega2 - params.central_freq - 2*(params.omega_M2 - comb_omega)) ** 2 + params.gamma ** 2)).sum(axis=1)


    def plot_all_modulations(mol, axes, clr):
        all_modulations = list(
            product(*(2 * [[params.omega_M1, params.omega_M2]]))
        )

        pol2 = np.zeros((4, params.freq.size), dtype=np.complex)
        for i, modulations in enumerate(all_modulations):
            if (i == 1) or (i == 2):
                print i, modulations
                pol2[i] = get_polarization2(mol, params, modulations)

        pol2 /= np.abs(pol2).max()
        pol2_sum_field_free = pol2[1] + pol2[2]
        pol2_sum_field_free /= np.abs(pol2_sum_field_free).max()
        comb_plot(frequency, pol2_sum_field_free.real, axes, clr, linewidth=2.)
        # axes.plot(frequency, pol2_sum_field_free.real, clr, linewidth=2.)
        return pol2_sum_field_free

    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # molecule.energies = np.cumsum([0, 2.01e5, 2.01e5])
    # plot_all_modulations(molecule, axes, 'k')
    # molecule.energies = np.cumsum([0, 2.009e5, 2.008e5])
    # plot_all_modulations(molecule, axes, 'r')
    # molecule.energies = np.cumsum([0, 2.007e5, 2.007e5])
    # plot_all_modulations(molecule, axes, 'b')
    # comb_plot(field_frequency1, field1/field1.max(), axes, 'r', alpha=0.5)
    # comb_plot(field_frequency2, field2/field2.max(), axes, 'b', alpha=0.5)
    # plt.show()

    def plot_no_modulations(ax1, clr):
        pol2 = get_polarization2(molecule, params, [params.omega_M2, params.omega_M1])
        ax1.plot(frequency / delta_freq, np.abs(pol2), color=clr, linewidth=2.)
        ax1.set_ylabel('$P^{(3)}(\\omega)$', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
        print time.time() - start

    def plot_spacing_dependence():
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        list_delta_freq = np.linspace(10, 15, 6).tolist()
        mol_energies = [np.cumsum([0, 2.01e5, 2.01e5])]
        colors = ['g', 'b', 'r']
        for i, delta in enumerate(list_delta_freq):
            axes[i/3, i % 3].set_title('$\Delta \omega = $' + str(list_delta_freq[i]))
            params.delta_freq = delta
            # params['freq'] = nonuniform_frequency_range_3(molecule, params)[0]
            for j, energy in enumerate(mol_energies):
                molecule.energies = energy
                plot_all_modulations(molecule, axes[i/3, i % 3], colors[j])

    plot_spacing_dependence()

    def plot_L_spectra_NL_pol2(molecule, ax, clr):
        spectra = linear_spectra(molecule, frequency)
        ax.plot(frequency, spectra, clr, linewidth=2.)
        # pol3 = get_polarization3(molecule, params, [params.omega_M2, params.omega_M2, params.omega_M1])
        # ax.plot(frequency, pol3.real, clr, linewidth=1., alpha=0.6)


    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    # fig1, axes1 = plt.subplots(nrows=1, ncols=1)
    # fig.suptitle("Total $P^{(3)}(\\omega)$")
    # molecule.energies = np.cumsum([0, 2.01e5, 2.01e5])
    # plot_L_spectra_NL_pol2(molecule, axes1, 'k')
    # molecule.energies = np.cumsum([0, 2.009e5, 2.008e5])
    # plot_L_spectra_NL_pol2(molecule, axes1, 'r')
    # molecule.energies = np.cumsum([0, 2.007e5, 2.007e5])
    # plot_L_spectra_NL_pol2(molecule, axes1, 'b')
    # axes.set_xlabel('$(\omega - \omega_{central})/ \Delta \omega$', color='k')
    # axes.set_ylabel('Field-free polarizations \n' + '$P^{(3)}(\\omega)$', color='k')
    # axes.tick_params('y', colors='k')
    # print time.time() - start
    # plt.show()

    # print frequency.max()/delta_freq, frequency.min()/delta_freq
    #
    # with open("pol3_matrix.pickle", "wb") as f:
    #     pickle.dump(
    #         {
    #             'pol3_matrix': pol2_matrix,
    #             'freq': frequency/delta_freq
    #         }, f
    #
    #     )

    frequency = np.linspace(0e5, 4.5e5, 5000)
    calculate_chi2(molecule, frequency, frequency)

