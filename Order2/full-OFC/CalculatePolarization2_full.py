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
    N_res = 11

    w0_pol3 = (omega_M1 + omega_M2) + params.central_freq
    w0_field1 = 2*omega_M1 + params.central_freq
    w0_field2 = 2*omega_M2 + params.central_freq

    freq_pol3 = w0_pol3 + np.linspace(-N*params.delta_freq, N*params.delta_freq, 2*N+1)[:, np.newaxis] + np.linspace(-0.3 * params.delta_freq, 0.3 * params.delta_freq, N_res)
    freq_field1 = w0_field1 + np.linspace(-N*params.delta_freq, N*params.delta_freq, 2*N+1)[:, np.newaxis] + np.linspace(-0.3 * params.delta_freq, 0.3 * params.delta_freq, N_res)
    freq_field2 = w0_field2 + np.linspace(-N*params.delta_freq, N*params.delta_freq, 2*N+1)[:, np.newaxis] + np.linspace(-0.3 * params.delta_freq, 0.3 * params.delta_freq, N_res)

    return freq_pol3.flatten(), freq_field1.flatten(), freq_field2.flatten()


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
    polarization = np.zeros(params.frequency.size, dtype=np.complex)
    polarization_mn = np.zeros_like(polarization, dtype=np.complex)

    for m, n in permutations(range(1, len(energy)), 2):
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
    ax.plot(frequency, value, *args, **kwargs)
    ax.get_xaxis().set_tick_params(which='both', direction='in', width=1)
    ax.get_yaxis().set_tick_params(which='both', direction='in', width=1)
    ax.get_xaxis().set_ticks_position('both')
    ax.get_yaxis().set_ticks_position('both')
    ax.grid(color='b', linestyle=':', linewidth=0.5)


def linear_spectra(molecule, omega):
    """
    Calculates linear spectra from linear susceptibility and molecular parameters
    :param molecule: 
    :return: 
    """
    transition = molecule.transitions
    energy = molecule.energies
    print(transition)
    print(energy)

    spectra = np.sum([transition[(n, 0)].mu**2*(energy[n] - energy[0])*transition[(n, 0)].g
                      / ((energy[n] - energy[0] - omega)**2 + transition[(n, 0)].g**2) for n in range(1, len(energy))]
                     , axis=0)
    print(spectra)
    return spectra


def calculate_chi2(molecule, omega, omega1):
    assert len(omega) == len(omega1), "Frequency arrays of unequal lengths"

    N = len(omega)
    transition = molecule.transitions
    energy = molecule.energies
    chi2 = np.empty((N, N), dtype=np.complex)

    for m, n in permutations((1, 2), 2):
        # print m, n
        chi2 += transition[(0, n)].mu * transition[(n, m)].mu * transition[(m, 0)].mu /\
                ((energy[n] - energy[0] - omega[:, np.newaxis] - 1j*transition[(n, 0)].g)
                 * (energy[m] - energy[0] - omega1[np.newaxis, :] - 1j*transition[(m, 0)].g))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    img = ax.imshow(chi2.real, extent=[omega.min(), omega.max(), omega1.min(), omega1.max()])
    fig.colorbar(img, ax=ax)


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
            (0, 1): CTransition(2.25e0, 1.),
            (1, 0): CTransition(2.25e0, 1.),
            (0, 2): CTransition(2.50e0, 1.),
            (2, 0): CTransition(2.50e0, 1.),
            (1, 2): CTransition(2.75e0, 1.),
            (2, 1): CTransition(2.75e0, 1.),
        }
    )

    params = ADict(
        central_freq=molecule.energies[2],
        comb_size=100,
        omega_M1=3.3,
        omega_M2=7.2,
        gamma=5e-6,
        delta_freq=delta_freq,
        width_g=4.,
        N_terms=10
    )

    import time
    start = time.time()

    def get_frequencies(molecule, params):
        params['frequency'] = nonuniform_frequency_range_3(molecule, params)[0]
        params['field_frequency1'] = nonuniform_frequency_range_3(molecule, params)[1]
        params['field_frequency2'] = nonuniform_frequency_range_3(molecule, params)[2]
        omega1 = params.field_frequency1[:, np.newaxis]
        omega2 = params.field_frequency2[:, np.newaxis]
        comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
        params['field1'] = (params.gamma / ((omega1 - params.central_freq - 2*(params.omega_M1 - comb_omega)) ** 2 + params.gamma ** 2)).sum(axis=1)
        params['field2'] = (params.gamma / ((omega2 - params.central_freq - 2*(params.omega_M2 - comb_omega)) ** 2 + params.gamma ** 2)).sum(axis=1)


    def plot_all_modulations(molecule, params, axes, axes_field, clr):
        get_frequencies(molecule, params)
        all_modulations = list(
            product(*(2 * [[params.omega_M1, params.omega_M2]]))
        )

        pol2 = np.zeros((4, params.frequency.size), dtype=np.complex)
        for i, modulations in enumerate(all_modulations):
            # if (i == 1) or (i == 2):
                # print(i, modulations)
            pol2[i] = get_polarization2(molecule, params, modulations)

        # pol2 /= np.abs(pol2).max()
        pol2_sum_field_free = pol2[1] + pol2[2]
        # pol2_sum_field_free /= np.abs(pol2_sum_field_free).max()
        comb_plot(params.frequency, pol2_sum_field_free.real, axes, clr, linewidth=2.)
        # comb_plot(params.field_frequency1, params.field1, axes_field, 'k', linewidth=1.)
        # comb_plot(params.field_frequency2, params.field2, axes_field, 'b', linewidth=1.)
        return pol2_sum_field_free


    def plot_no_modulations(ax1, clr):
        print(params.delta_freq)
        pol2 = get_polarization2(molecule, params, [params.omega_M2, params.omega_M1])
        comb_plot(params.frequency, pol2.real, ax1, color=clr, linewidth=2.)
        ax1.set_ylabel('$P^{(2)}(\\omega)$', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_xlabel("$\\omega_1 + \\omega_2 + \\Delta \\omega$ (in GHz)")
        print(time.time() - start)

    def plot_spacing_dependence():
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        list_delta_freq = np.linspace(10, 15, 6).tolist()
        mol_energies = [np.cumsum([0, 2.01e5, 2.01e5])]
        colors = ['g', 'b', 'r']
        for i, delta in enumerate(list_delta_freq):
            axes[int(i/3), int(i % 3)].set_title('$\Delta \omega = $' + str(list_delta_freq[i]))
            params.delta_freq = delta
            get_frequencies(molecule, params)
            for j, energy in enumerate(mol_energies):
                molecule.energies = energy
                plot_all_modulations(molecule, params, axes[int(i/3), int(i % 3)], colors[j])

    # plot_spacing_dependence()

    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # get_frequencies(molecule, params)
    # plot_all_modulations(molecule, params, axes, 'r')

    def plot_L_spectra_NL_pol2(molecule, params, ax, clr):
        spectra = linear_spectra(molecule, params.frequency)
        ax.plot(params.frequency, spectra, clr, linewidth=2.)
        # pol3 = get_polarization3(molecule, params, [params.omega_M2, params.omega_M2, params.omega_M1])
        # ax.plot(frequency, pol3.real, clr, linewidth=1., alpha=0.6)


    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes_field = axes.twinx()
    fig.suptitle("Total $P^{(3)}(\\omega)$")
    molecule.energies = np.cumsum([0, 2.01e5, 2.01e5])
    plot_all_modulations(molecule, params, axes, axes_field, 'r')
    # plot_L_spectra_NL_pol2(molecule, params, axes1, 'k')
    # molecule.energies = np.cumsum([0, 2.009e5, 2.008e5])
    # plot_L_spectra_NL_pol2(molecule, axes1, 'r')
    # molecule.energies = np.cumsum([0, 2.007e5, 2.007e5])
    # plot_L_spectra_NL_pol2(molecule, axes1, 'b')
    axes.set_xlabel('$(\omega - \omega_{central})/ \Delta \omega$', color='k')
    axes.set_ylabel('Field-free polarizations \n' + '$P^{(3)}(\\omega)$', color='k')
    axes.tick_params('y', colors='k')
    print(time.time() - start)
    plt.show()

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

    # frequency = np.linspace(0e5, 4.5e5, 5000)
    # calculate_chi2(molecule, frequency, frequency)
    # plt.show()


