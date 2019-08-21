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
    freq_central = molecule.energies[0]

    # w0 = np.ceil((energy[len(energy)-1] - energy[0])/params.delta_freq)
    # w0_pol3 = np.ceil((energy[len(energy)-2] - energy[len(energy)-3])/params.delta_freq) - (2 * omega_M2 - omega_M1)
    w0_pol3_21 = freq_central + (2 * omega_M2 - omega_M1)
    w0_pol3_12 = freq_central + (2 * omega_M1 - omega_M2)
    w0_field1 = freq_central + omega_M1
    w0_field2 = freq_central + omega_M2

    N_res = 11
    N_comb = params.comb_size

    freq_points = np.linspace(-N_comb * params.delta_freq, N_comb * params.delta_freq, N_comb + 1)[:,
                                     np.newaxis]
    resolution = np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, N_res)

    frequency_12 = w0_pol3_12 + np.linspace(-N_comb * params.delta_freq, N_comb * params.delta_freq, N_comb + 1)[:,
                                     np.newaxis] + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, N_res)
    frequency_21 = w0_pol3_21 + np.linspace(-N_comb * params.delta_freq, N_comb * params.delta_freq, N_comb + 1)[:,
                                     np.newaxis] + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, N_res)
    field_freq1 = w0_field1 + np.linspace(-N_comb * params.delta_freq, N_comb * params.delta_freq, 2 * N_comb + 1)[:,
                                   np.newaxis] + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, N_res)
    field_freq2 = w0_field2 + np.linspace(-N_comb * params.delta_freq, N_comb * params.delta_freq, 2 * N_comb + 1)[:,
                                   np.newaxis] + np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, N_res)

    freq_pol3 = np.sort(np.hstack([frequency_12.flatten(), frequency_21.flatten()]))
    freq_field1 = field_freq1.flatten()
    freq_field2 = field_freq2.flatten()

    return np.ascontiguousarray(freq_pol3), np.ascontiguousarray(freq_field1), np.ascontiguousarray(freq_field2)


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

    for i, modulations in enumerate(list(product(*(3 * [[params.omega_M1, params.omega_M2]])))):
        for m, n, v in permutations(range(1, len(energy)), 3):
            try:
                # calculate the product of the transition dipole if they are not zeros
                mu_product = transition[(0, m)].mu * transition[(m, n)].mu * \
                             transition[(n, v)].mu * transition[(v, 0)].mu

                # reset the polarization because C-code performs "+="
                polarization_mnv[:] = 0.
                print(modulations, m, n, v)

                pol3_total(
                    polarization_mnv, params,
                    modulations[0], modulations[1], modulations[2],
                    energy[n] - energy[v] - 1j * transition[(n, v)].g,
                    energy[m] - energy[v] - 1j * transition[(m, v)].g,
                    energy[v] - energy[0] - 1j * transition[(v, 0)].g,
                    energy[n] - energy[0] - 1j * transition[(n, 0)].g,
                    energy[m] - energy[0] - 1j * transition[(m, 0)].g,
                    energy[m] - energy[n] - 1j * transition[(m, n)].g,
                    energy[n] - energy[m] - 1j * transition[(n, m)].g,
                    energy[v] - energy[n] - 1j * transition[(v, n)].g,
                    energy[v] - energy[m] - 1j * transition[(v, m)].g,

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
    print(spectra)
    return spectra


#################
#               #
#   Run test    #
#               #
#################


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    # UNIT OF FREQUENCY IS 10 MHz

    freq_unit = 1/(1000 / 0.024188)

    # dipole value and line width for each transition
    # population decay rates (Non-zero for relaxation from higher to lower energy)
    gamma_decay = np.ones((4, 4)) * 1e-3 * freq_unit  # All population relaxation times equal 1 GHz (1 ns inverse)
    np.fill_diagonal(gamma_decay, 0.0)    # Diagonal elements zero; no decay to self
    gamma_decay = np.tril(gamma_decay)    # Relaxation only to lower energy states
    # dephasing rates (T_ij = T_ji for dephasing)
    gamma_dephasing = np.ones((4, 4)) * 1e1 * freq_unit     # All electronic dephasing rates are 10 THz (100 fs inverse)
    np.fill_diagonal(gamma_dephasing, 0.0)
    gamma_dephasing[0, 1] = 0.59e0 * freq_unit           # All vibrational dephasing rates are 0.59 THz (1.7 ps inverse)
    gamma_dephasing[1, 0] = 0.59e0 * freq_unit
    gamma_dephasing[2, 3] = 0.59e0 * freq_unit
    gamma_dephasing[3, 2] = 0.59e0 * freq_unit

    # Net damping rates given by Boyd pg. 156, G_nm = (1/2) * \sum_i (g_decay_ni + g_decay_mi) + g_dephasing_nm

    gamma = np.zeros_like(gamma_decay)
    for n in range(4):
        for m in range(4):
            for i in range(4):
                gamma[n, m] += 0.5 * (gamma_decay[n, i] + gamma_decay[m, i])
            gamma[n, m] += gamma_dephasing[n, m]

    # gamma *= 1e-2

    print(gamma)
    molecule = ADict(

        #  Energy difference of levels (should be multiples of delta_freq)
        energies=np.cumsum([0, 48, 500, 45]) * freq_unit,      # Energies are cumsum(113, 2844, 113) in THz
        # Nile Blue parameters -> 1->2 = 3->4 = 600 cm-1; 1->3 = 15700 cm-1

        transitions={
            (0, 1): CTransition(gamma[0, 1], 2),
            (1, 0): CTransition(gamma[1, 0], 2),
            (0, 2): CTransition(gamma[0, 2], 2),
            (2, 0): CTransition(gamma[2, 0], 2),
            (0, 3): CTransition(gamma[0, 3], 2),
            (3, 0): CTransition(gamma[3, 0], 2),
            (1, 2): CTransition(gamma[1, 2], 2),
            (2, 1): CTransition(gamma[2, 1], 2),
            (1, 3): CTransition(gamma[1, 3], 2),
            (3, 1): CTransition(gamma[3, 1], 2),
            (2, 3): CTransition(gamma[2, 3], 2),
            (3, 2): CTransition(gamma[3, 2], 2)
        }
    )

    params = ADict(
        comb_size=100,
        omega_M1=6e-1 * freq_unit,
        omega_M2=2e-1 * freq_unit,
        gamma=1e-12 * freq_unit,
        delta_freq=1e-0 * freq_unit,
        width_g=6.,
        N_terms=5
    )

    import time
    start = time.time()
    frequency = nonuniform_frequency_range_3(molecule, params)[0]
    params['freq'] = frequency

    print(time.time() - start)

    # field_frequency = params.delta_freq*np.linspace(-params.comb_size, params.comb_size, params.N_frequency)
    field_frequency1 = nonuniform_frequency_range_3(molecule, params)[1]
    field_frequency2 = nonuniform_frequency_range_3(molecule, params)[2]
    omega1 = field_frequency1[:, np.newaxis]
    omega2 = field_frequency2[:, np.newaxis]
    comb_omega = (params.delta_freq * np.arange(-params.comb_size, params.comb_size))[np.newaxis, :]
    field1 = (params.gamma / ((omega1 - params.omega_M1 - comb_omega) ** 2 + params.gamma ** 2)).sum(axis=1)
    field2 = (params.gamma / ((omega2 - params.omega_M2 - comb_omega) ** 2 + params.gamma ** 2)).sum(axis=1)

    def plot_all_modulations(axes, clr, mod_num):
        pol3 = np.zeros(params.freq.size, dtype=np.complex)
        pol3 += get_polarization3(molecule, params)

        print(pol3.max(), '\n')

        pol3_sum_field_free = pol3
        comb_plot(frequency / params.delta_freq, -pol3_sum_field_free.real, axes[0], clr, linewidth=1.)
        comb_plot(frequency / params.delta_freq, -pol3_sum_field_free.imag, axes[1], clr, linewidth=1.)
        comb_plot(field_frequency1 / params.delta_freq, field1 * pol3_sum_field_free.real.max() / field1.max(), axes[0], 'y', alpha=0.4)
        comb_plot(field_frequency2 / params.delta_freq, field2 * pol3_sum_field_free.real.max() / field1.max(), axes[0], 'g', alpha=0.4)

        print(field1.size)
        return pol3_sum_field_free

    # def plot_no_modulations(ax1, clr):
    #     pol3 = get_polarization3(molecule, params, [params.omega_M2, params.omega_M2, params.omega_M1])
    #     ax1.plot(frequency / delta_freq, np.abs(pol3), color=clr, linewidth=2.)
    #     ax1.set_ylabel('$P^{(3)}(\\omega)$', color='k')
    #     ax1.tick_params('y', colors='k')
    #     ax1.set_xlabel("$\\omega_1 + \\omega_2 - \\omega_3 + \\Delta \\omega$ (in GHz)")
    #     print(time.time() - start)

    # def plot_spacing_dependence():
    #     fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    #     list_delta_freq = np.linspace(0.1, 0.60, 6).tolist()
    #     mol_energies = [np.cumsum([0, 3, 2.01e3, 3]) * delta_freq, np.cumsum([0, 3, 2.01e3, 4]) * delta_freq, np.cumsum([0, 4, 2.01e3, 3]) * delta_freq]
    #     colors = ['r', 'b', 'g']
    #     for i, delta in enumerate(list_delta_freq):
    #         axes[i/3, i % 3].set_title('$\Delta \omega = $' + str(list_delta_freq[i]))
    #         params.delta_freq = delta
    #         for j, energy in enumerate(mol_energies):
    #             molecule.energies = energy
    #             plot_no_modulations(axes[i/3, i % 3], colors[j])

    # plot_spacing_dependence()

    # def plot_L_spectra_NL_pol3(molecule, ax, clr):
    #     frequency = np.linspace(3.5e6 - 10, 3.5e6 + 15, 10000)
    #     spectra = linear_spectra(molecule, frequency)
    #     ax.plot(frequency, spectra, clr, linewidth=2.)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    # fig1, axes1 = plt.subplots(nrows=1, ncols=1)

    fig.suptitle("Total $P^{(3)}(\\omega)$ for Nile Blue")
    # pol3_matrix = np.empty((params.freq.size, 8), dtype=np.complex)

    # molecule.energies = np.cumsum([0, 113, 2844, 113])
    # plot_L_spectra_NL_pol3(molecule, axes1, 'k')
    # comb_plot(field_frequency1 / delta_freq, field1.real * 4.25e6 / field1.max(), axes, 'c', linewidth=1., alpha=0.5)
    # comb_plot(field_frequency2 / delta_freq, field2.real * 4.25e6 / field2.max(), axes, 'b', linewidth=1., alpha=0.5)
    plot_all_modulations(axes, 'r', 1)
    # pol3_matrix[:, 0] = plot_all_modulations(axes, 'k', 0)
    # pol3_matrix[:, 2] = plot_all_modulations(axes, 'r', 2)
    # pol3_matrix[:, 3] = plot_all_modulations(axes, 'r-.', 3)
    # pol3_matrix[:, 4] = plot_all_modulations(axes, 'b', 4)
    # pol3_matrix[:, 5] = plot_all_modulations(axes, 'b-.', 5)
    # pol3_matrix[:, 7] = plot_all_modulations(axes, 'm-.', 7)

    # molecule.energies = np.cumsum([0, 25, 3.5e7, 36]) * delta_freq
    # plot_L_spectra_NL_pol3(molecule, axes1, 'b')
    # pol3_matrix[:, 1] = plot_all_modulations(axes, 'b')

    # molecule.energies = np.cumsum([0, 24, 3.5e7, 37]) * delta_freq
    # plot_L_spectra_NL_pol3(molecule, axes1, 'r')
    # pol3_matrix[:, 2] = plot_all_modulations(axes, 'r')

    axes[0].set_xlabel('$(\\omega - \\omega_{central})/ \\Delta \\omega$', color='k')
    axes[0].set_ylabel('Field-free polarizations \n' + '$P^{(3)}(\\omega)$', color='k')
    axes[0].tick_params('y', colors='k')
    print(time.time() - start)

    plt.show()

    # with open("pol3_matrix.pickle", "wb") as f:
    #     pickle.dump(
    #         {
    #             'pol3_matrix': pol3_matrix,
    #             'freq': frequency/delta_freq
    #         }, f
    #
    #     )


