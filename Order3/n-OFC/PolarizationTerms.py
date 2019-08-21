import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType
import numexpr as ne


class PolarizationTerms:
    """
    Calculates NL Polarizations for an ensemble of near identical molecules.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified:

        N_molecules: NUMBER OF MOLECULES IN THE DISCRIMINATION PROBLEM
        N_comb: NUMBER OF COMB LINES USED IN THE CALCULATION
        N_frequency: NUMBER OF FREQUENCY POINTS DESCRIBING POLARIZATION AND FIELDS

        ---------------------------------------------------
        THE FOLLOWING PARAMETERS ALL HAVE THE UNITS OF fs-1
        ---------------------------------------------------

        freq_halfwidth: HALF LENGTH OF FREQUENCY ARRAY CENTERED AROUND w_excited_1
            s.t. FREQ = [w_excited_1 - freq_halfwidth --> w_excited_1 + freq_halfwidth]

        w_excited_1: omega_10 = (1/hbar) * (E_1 - E_0)
        w_excited_2: omega_20 = (1/hbar) * (E_2 - E_0)

        omega_M1: MODULATION FREQUENCY 1 = omega_10 + OFFSET_1
        omega_M2: MODULATION FREQUENCY 2 = omega_10 + OFFSET_2
        gamma: ELECTRIC FIELD LINE-WIDTH
        omega_del_1: COMB-SPACING FOR MODULATOR 1
        omega_del_2: COMB-SPACING FOR MODULATOR 2

        w_spacing_10: omega_10 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        w_spacing_20: omega_20 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        g_spacing_10: gamma_10 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        g_spacing_12: gamma_12 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        g_spacing_20: gamma_20 INCREMENT (in GHz) BETWEEN TWO CONSECUTIVE MOLECULES
        """

        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.N_molecules
        except AttributeError:
            raise AttributeError("Number of molecules not specified")

        try:
            self.N_comb
        except AttributeError:
            raise AttributeError("Number of comb lines not specified")

        try:
            self.N_frequency
        except AttributeError:
            raise AttributeError("Number of frequency points not specified")

        try:
            self.w_excited_1
        except AttributeError:
            raise AttributeError("omega_10 not specified")

        try:
            self.w_excited_2
        except AttributeError:
            raise AttributeError("omega_20 not specified")

        try:
            self.omega_M1
        except AttributeError:
            raise AttributeError("Modulation frequency 1 not specified")

        try:
            self.omega_M2
        except AttributeError:
            raise AttributeError("Modulation frequency 2 not specified")

        try:
            self.gamma
        except AttributeError:
            raise AttributeError("Field line-width not specified")

        try:
            self.omega_del_1
        except AttributeError:
            raise AttributeError("Frequency Comb spacing 1 not specified")

        try:
            self.omega_del_2
        except AttributeError:
            raise AttributeError("Frequency Comb spacing 2 not specified")

        try:
            self.w_spacing_10
        except AttributeError:
            raise AttributeError("Molecule omega_10 increments not specified")

        try:
            self.g_spacing_10
        except AttributeError:
            raise AttributeError("Molecule gamma_10 increments not specified")

        try:
            self.w_spacing_20
        except AttributeError:
            raise AttributeError("Molecule omega_20 increments not specified")

        try:
            self.g_spacing_20
        except AttributeError:
            raise AttributeError("Molecule gamma_20 increments not specified")

        try:
            self.g_spacing_12
        except AttributeError:
            raise AttributeError("Molecule gamma_12 increments not specified")

        self.molecules = [
            dict(
                w_10=2.354 + _ * self.w_spacing_10 * 10**(-self.N_order_energy),
                g_10=_ * self.g_spacing_10 * 10**(-self.N_order),
                w_20=4.708 + _ * self.w_spacing_20 * 10**(-self.N_order_energy),
                g_20=_ * self.g_spacing_20 * 10**(-self.N_order),
                g_12=_ * self.g_spacing_12 * 10**(-self.N_order),
            ) for _ in range(1, self.N_molecules+1)
        ]

        [self.molecules[_].update(
            {
                'w_12': self.molecules[_]['w_20'] - self.molecules[_]['w_10']
            }
        ) for _ in range(self.N_molecules)]

        [self.molecules[_].update(
            {
                'w_21': self.molecules[_]['w_12'],
                'g_21': self.molecules[_]['g_12']

            }
        ) for _ in range(self.N_molecules)]

        self.frequency = np.linspace(
            2.*self.w_excited_1 - self.freq_halfwidth,
            2.*self.w_excited_1 + self.freq_halfwidth - 2*self.freq_halfwidth/self.N_frequency,
            self.N_frequency
        )

        self.del_omega1 = self.omega_del_1 * np.arange(-self.N_comb, self.N_comb)
        self.del_omega2 = self.omega_del_2 * np.arange(-self.N_comb, self.N_comb)

        gaussian = np.exp(-(np.arange(-self.N_comb, self.N_comb))**2/(.04*self.N_comb**2))
        self.omega = self.frequency[:, np.newaxis, np.newaxis]
        self.comb_omega1 = self.del_omega1[np.newaxis, :, np.newaxis]
        self.comb_omega2 = self.del_omega2[np.newaxis, np.newaxis, :]
        self.shape_1 = gaussian[np.newaxis, :, np.newaxis]
        self.shape_2 = gaussian[np.newaxis, np.newaxis, :]

        self.field1 = ne.evaluate(
            "sum(shape_1*gamma / ((omega - 2*omega_M1 - 2*comb_omega1)**2 + gamma**2), axis=2)",
            local_dict=vars(self)
        ).sum(axis=1)
        self.field2 = ne.evaluate(
            "sum(shape_2*gamma / ((omega - 2*omega_M2 - 2*comb_omega2)**2 + gamma**2), axis=2)",
            local_dict=vars(self)
        ).sum(axis=1)

        # shape1 = np.random.uniform(0.5, 1., self.field1.size)*gaussian
        # shape2 = np.random.uniform(0.5, 1., self.field1.size)*gaussian
        #
        # shape1 = np.ones_like(shape1)
        # shape2 = np.ones_like(shape2)
        #
        # self.field1 *= shape1
        # self.field2 *= shape2

        # plt.figure()
        # plt.subplot(211)
        # plt.plot(self.frequency-2.*self.w_excited_1, self.field1, 'b')
        # plt.subplot(212)
        # plt.plot(self.frequency-2.*self.w_excited_1, self.field2, 'r')
        # plt.show()
        # self.pol2_freq_matrix = np.asarray([self.calculate_total_pol(**instance) for instance in self.molecules]).T

    def calculate_pol_12_21(self, m, n, **params):
        """
        CALCULATES THE POLARIZATION DUE TO THE TERM (a2') OF THE SUSCEPTIBILITY TENSOR, INTERACTING WITH THE COMPONENTS
        E_1(omega_1) AND E_2(omega - omega_1) and E_1(omega - omega_1) AND E_2(omega_1)
        :param params: FREQUENCY AND LINE-WIDTHS REQUIRED TO DEFINE A THREE-LEVEL MOLECULAR SYSTEM: w_10, w_20, w_12,
        w_21, g_10, g_20, g_12, g_21
        :return: P^(2)(omega)_(a1)
        """
        term_J1 = np.pi*self.gamma*(self.omega - 2*params['w_'+str(m)+str(0)] + self.omega_M1 - self.omega_M2 + self.comb_omega1 - self.comb_omega2)
        term_J2 = np.pi*self.gamma*(self.omega - 2*params['w_'+str(m)+str(0)] + self.omega_M2 - self.omega_M1 + self.comb_omega2 - self.comb_omega1)

        term_K1 = (self.omega - self.omega_M2 - self.comb_omega2 - params['w_'+str(m)+str(0)]) + 1j*(self.gamma + params['g_'+str(m)+str(0)])
        term_K2 = (self.omega_M1 + self.comb_omega1 - params['w_'+str(m)+str(0)]) + 1j*(self.gamma + params['g_'+str(m)+str(0)])

        term_K1_d = (self.omega - self.omega_M1 - self.comb_omega1 - params['w_'+str(m)+str(0)]) + 1j * (self.gamma + params['g_'+str(m)+str(0)])
        term_K2_d = (self.omega_M2 + self.comb_omega2 - params['w_'+str(m)+str(0)]) + 1j * (self.gamma + params['g_'+str(m)+str(0)])

        delta = (self.omega - self.omega_M1 - self.omega_M2 - self.comb_omega1 - self.comb_omega2)**2 + 4.*self.gamma**2

        A_012 = params['w_'+str(n)+str(0)] - self.omega - 1j*params['g_'+str(n)+str(0)]
        return ne.evaluate(
            "sum((term_J1/(term_K1*term_K2) + term_J2/(term_K1_d*term_K2_d))/(delta*A_012), axis=2)"
        ).sum(axis=1)

        # return (np.pi*self.gamma/delta).sum(axis=(1, 2))

    def calculate_pol_12_21_a1_a2(self, m, n, **params):
        """
        CALCULATES THE POLARIZATION DUE TO THE TERM (a2') OF THE SUSCEPTIBILITY TENSOR, INTERACTING WITH THE COMPONENTS
        E_1(omega_1) AND E_2(omega - omega_1) and E_1(omega - omega_1) AND E_2(omega_1)
        :param params: FREQUENCY AND LINE-WIDTHS REQUIRED TO DEFINE A THREE-LEVEL MOLECULAR SYSTEM: w_10, w_20, w_12,
        w_21, g_10, g_20, g_12, g_21
        :return: P^(2)(omega)_(a1)
        """
        term_I = 4j*np.pi*self.gamma*(self.gamma + 0.5*params['g_'+str(m)+str(0)])
        term_J1 = np.pi * self.gamma * (self.omega - 2 * params['w_' + str(m) + str(0)] + self.omega_M1 - self.omega_M2 + self.comb_omega1 - self.comb_omega2)
        term_J2 = np.pi * self.gamma * (self.omega - 2 * params['w_' + str(m) + str(0)] + self.omega_M2 - self.omega_M1 + self.comb_omega2 - self.comb_omega1)
        term_K1 = (self.omega - self.omega_M2 - self.comb_omega2 - params['w_' + str(m) + str(0)]) + 1j * (self.gamma + params['g_' + str(m) + str(0)])
        term_K2 = (self.omega_M1 + self.comb_omega1 - params['w_' + str(m) + str(0)]) + 1j * (self.gamma + params['g_' + str(m) + str(0)])
        term_K1_d = (self.omega - self.omega_M1 - self.comb_omega1 - params['w_' + str(m) + str(0)]) + 1j * (self.gamma + params['g_' + str(m) + str(0)])
        term_K2_d = (self.omega_M2 + self.comb_omega2 - params['w_' + str(m) + str(0)]) + 1j * (self.gamma + params['g_' + str(m) + str(0)])
        delta = (self.omega - self.omega_M1 - self.omega_M2 - self.comb_omega1 - self.comb_omega2) ** 2 + 4. * self.gamma ** 2
        A_012 = params['w_' + str(n) + str(0)] - self.omega - 1j * params['g_' + str(n) + str(0)]

        shape_1 = self.shape_1
        shape_2 = self.shape_2

        return ne.evaluate(
            "sum(shape_1*shape_2*(((term_J1 + term_I)/(term_K1*term_K2)) + ((term_J2 - term_I)/(term_K1_d*term_K2_d)))/(delta*A_012), axis=2)"
        ).sum(axis=1), ne.evaluate(
            "sum(shape_1*shape_2*(((term_J1 - term_I)/(term_K1*term_K2)) + ((term_J2 + term_I)/(term_K1_d*term_K2_d)))/(delta*A_012), axis=2)"
        ).sum(axis=1)

    def calculate_pol_11_22_a1_a2(self, m, n, **params):
        """
        CALCULATES THE POLARIZATION DUE TO THE TERMS (a1) and (a2) OF THE SUSCEPTIBILITY TENSOR, INTERACTING WITH THE COMPONENTS
        E_1(omega_1) AND E_2(omega - omega_1) and E_1(omega - omega_1) AND E_2(omega_1)
        :param params: FREQUENCY AND LINE-WIDTHS REQUIRED TO DEFINE A THREE-LEVEL MOLECULAR SYSTEM: w_10, w_20, w_12,
        w_21, g_10, g_20, g_12, g_21
        :return: P^(2)(omega)_(a1)
        """
        term_I = 4j*np.pi*self.gamma*(self.gamma + 0.5*params['g_'+str(m)+str(0)])
        term_J = np.pi * self.gamma * (self.omega - 2 * params['w_' + str(m) + str(0)])

        term_K1 = (self.omega - self.omega_M1 - self.comb_omega1 - params['w_' + str(m) + str(0)]) + 1j * (self.gamma + params['g_' + str(m) + str(0)])
        term_K2 = (self.omega_M1 + self.comb_omega1 - params['w_' + str(m) + str(0)]) + 1j * (self.gamma + params['g_' + str(m) + str(0)])

        term_K1_d = (self.omega - self.omega_M2 - self.comb_omega2 - params['w_' + str(m) + str(0)]) + 1j * (self.gamma + params['g_' + str(m) + str(0)])
        term_K2_d = (self.omega_M2 + self.comb_omega2 - params['w_' + str(m) + str(0)]) + 1j * (self.gamma + params['g_' + str(m) + str(0)])

        delta_1 = (self.omega - 2.*self.omega_M1 - 2.*self.comb_omega1) ** 2 + 4. * self.gamma ** 2
        delta_2 = (self.omega - 2.*self.omega_M2 - 2.*self.comb_omega2) ** 2 + 4. * self.gamma ** 2
        A_012 = params['w_' + str(n) + str(0)] - self.omega - 1j * params['g_' + str(n) + str(0)]

        shape_1 = self.shape_1
        shape_2 = self.shape_2
        return ne.evaluate(
            "sum(shape_1*shape_2*(2.*term_J/(term_K1*term_K2))/(delta_1*A_012), axis=2)"
        ).sum(axis=1), ne.evaluate(
            "sum(shape_1*shape_2*(2.*term_J/(term_K1_d*term_K2_d))/(delta_2*A_012), axis=2)"
        ).sum(axis=1)

    # def calculate_total_pol(self, **params):
    #     return self.calculate_pol_12_21(1, 2, **params) + self.calculate_pol_12_21(2, 1, **params)

    def calculate_chi(self, m, n, **params):
        return 1./((params['w_'+str(n)+str(0)] - self.frequency[:, np.newaxis]
                    - self.frequency[np.newaxis, :] - 1j*params['g_'+str(n)+str(0)])
                   *(params['w_'+str(m)+str(0)] - self.frequency[:, np.newaxis] - 1j*params['g_'+str(m)+str(0)])) \
               + 1./((params['w_'+str(n)+str(0)] - self.frequency[:, np.newaxis]
                      - self.frequency[np.newaxis, :] - 1j*params['g_'+str(n)+str(0)])
                     *(params['w_'+str(m)+str(0)] - self.frequency[np.newaxis, :] - 1j*params['g_'+str(m)+str(0)]))

    def calculate_broad_response(self):
        return 2.*self.gamma/((self.omega - self.omega_M1 - self.omega_M2 - self.comb_omega1 - self.comb_omega2)**2 + 4.*self.gamma**2)


if __name__ == '__main__':
    from scipy.interpolate import interp1d
    import math

    w_excited_1 = 2.354
    w_excited_2 = 4.708

    ensemble = PolarizationTerms(
        N_molecules=1,
        N_order=7,
        N_order_energy=9,
        N_comb=50,
        N_frequency=2000,
        freq_halfwidth=1.e-5,

        w_excited_1=w_excited_1,
        w_excited_2=w_excited_2,

        omega_M1=w_excited_1 + .5e-7,
        omega_M2=w_excited_1 + 1.5e-7,
        gamma=2.5e-9,
        omega_del_1=2.4e-7,
        omega_del_2=2.4e-7,

        w_spacing_10=6.0,
        w_spacing_20=7.0,
        g_spacing_10=25,
        g_spacing_12=10,
        g_spacing_20=15,
    )
    #
    # N_ = 80
    # pol_order = np.zeros(N_)
    # order = np.zeros(N_)
    # for i in range(N_):
    #     ensemble.gamma = 10**(-8-i/10.)
    #     order[i] = ensemble.gamma
    #     pol_order[i] = np.abs(ensemble.calculate_total_pol(**ensemble.molecules[0]).max())
    #     print i, order[i], pol_order[i]
    # plt.figure()
    # plt.suptitle("Dependence of $P^{(2)}(\omega)$ on the comb linewidth")
    # plt.plot(-np.log10(order), np.log10(pol_order))
    # plt.xlabel("$-log_{10}(\gamma)$")
    # plt.ylabel("$-log_{10}(|P^{(2)}(\omega)|)$")
    # plt.show()

    print(ensemble.molecules[0])
    ensemble.gamma = 1e-9
    # f, ax = plt.subplots(3, 3, sharex=True)
    # f.suptitle('$2^{nd}$ order nonlinear polarizations \n over a broad linewidth range (0.1Hz to 10MHz)')
    # colors = 'k' + 'b'*7 + 'k'
    # titles = ['10 MHz', '1MHz', '0.1MHz', '10 KHz', '1KHz', '0.1KHz', '10 Hz', '1Hz', '0.1Hz']
    # for i in range(3):
    #     for j in range(3):
    #         ensemble.gamma = 10**(-8-3*i-j)
    #         ax[i, j].plot(ensemble.frequency, ensemble.calculate_total_pol(**ensemble.molecules[0]).real, color=colors[3*i+j])
    #         print ensemble.gamma
    #         ax[i, j].set_title(titles[3*i+j])
            # for label in ax[i, j].get_xmajorticklabels() + ax[i, j].get_ymajorticklabels():
            #     label.set_rotation(30)
        # ax[i, 0].set_ylabel("Normalized Polarization \n (Real part) \n $\mathcal{Re}[P^{(2)}(\\omega)]$")
    #
    # ax[2, 0].set_xlabel("Frequency (in $fs$)")
    # ax[2, 1].set_xlabel("Frequency (in $fs$)")
    # ax[2, 2].set_xlabel("Frequency (in $fs$)")

    pol_diagonal_12 = ensemble.calculate_pol_11_22_a1_a2(1, 2, **ensemble.molecules[0])
    pol_diagonal_21 = ensemble.calculate_pol_11_22_a1_a2(2, 1, **ensemble.molecules[0])
    pol_off_diagonal_12 = ensemble.calculate_pol_12_21_a1_a2(1, 2, **ensemble.molecules[0])
    pol_off_diagonal_21 = ensemble.calculate_pol_12_21_a1_a2(2, 1, **ensemble.molecules[0])

    pol_11 = pol_diagonal_12[0] + pol_diagonal_21[0]
    pol_22 = pol_diagonal_12[1] + pol_diagonal_21[1]
    pol_12_21 = pol_off_diagonal_12[0] + pol_off_diagonal_21[0]

    factor12 = np.abs(ensemble.calculate_pol_12_21_a1_a2(1, 2, **ensemble.molecules[0])[0]).max() / ensemble.field1.max()
    factor11 = np.abs(ensemble.calculate_pol_11_22_a1_a2(1, 2, **ensemble.molecules[0])[0]).max() / ensemble.field1.max()

    frequency_axis = np.linspace(-1e-5, 1e-5, ensemble.frequency.size)
    f, ax = plt.subplots(2, 1, sharex=True)
    f.suptitle("Contribution of 4 symmetric terms in the imaginary part of the $2^{nd}$ order nonlinear polarization")
    ax[0].plot(frequency_axis, ensemble.field1/(factor12 / factor11)/4e8, 'r', alpha=0.4)
    ax[0].plot(frequency_axis, ensemble.field2/(factor12 / factor11)/4e8, 'b', alpha=0.4)
    ax[1].plot(frequency_axis, ensemble.field1 / 4e8, 'r', alpha=0.4)
    ax[1].plot(frequency_axis, ensemble.field2 / 4e8, 'b', alpha=0.4)
    #
    ax[0].plot(frequency_axis, pol_11.imag/factor12/4e8, 'firebrick', label='$\mathcal{Im}[P^{(2)}_{11}(\\omega)]$')
    ax[1].plot(frequency_axis, pol_12_21.imag/factor12/4e8, 'k', label='$\mathcal{Im}[P^{(2)}_{12}(\\omega) + P^{(2)}_{21}(\\omega)]$')
    ax[0].plot(frequency_axis, pol_22.imag/factor12/4e8, 'navy', label='$\mathcal{Im}[P^{(2)}_{22}(\\omega)]$')
    #
    # ax[0].plot(frequency_axis, pol_11.imag, 'firebrick', label='$\mathcal{Im}[P^{(2)}_{11}(\\omega)]$')
    # ax[1].plot(frequency_axis, pol_12_21.imag, 'k', label='$\mathcal{Im}[P^{(2)}_{12}(\\omega)]$')
    # ax[0].plot(frequency_axis, pol_22.imag, 'navy', label='$\mathcal{Im}[P^{(2)}_{22}(\\omega)]$')

    ax[0].legend(loc=1)
    ax[1].legend(loc=1)
    ax[0].set_ylabel("$\mathcal{Im}[P^{(2)}(\\omega)]$)")
    ax[1].set_ylabel("$\mathcal{Im}[P^{(2)}(\\omega)]$")
    ax[1].set_xlabel("$400 nm + \Delta \omega$ (in $GHz$)")
    ax[0].yaxis.set_label_position("right")
    ax[1].yaxis.set_label_position("right")
    # for i in range(3):
    #     for label in ax[i].get_xmajorticklabels() + ax[i].get_ymajorticklabels():
    #         label.set_rotation(30)

    f, ax = plt.subplots(2, 1, sharex=True)
    # f.suptitle("Contribution of 4 symmetric terms in the real part of the $2^{nd}$ order nonlinear polarization")

    ax[0].plot(frequency_axis, ensemble.field1 / (factor12 / factor11) / 4e8, 'r', alpha=0.4)
    ax[0].plot(frequency_axis, ensemble.field2 / (factor12 / factor11) / 4e8, 'b', alpha=0.4)
    ax[1].plot(frequency_axis, ensemble.field1 / 4e8, 'r', alpha=0.4)
    ax[1].plot(frequency_axis, ensemble.field2 / 4e8, 'b', alpha=0.4)

    ax[0].plot(frequency_axis, pol_11.real/factor12/4e8, 'firebrick', label='$\mathcal{Re}[P^{(2)}_{11}(\\omega)]$')
    ax[1].plot(frequency_axis, pol_12_21.real/factor12/4e8, 'k', label='$\mathcal{Re}[P^{(2)}_{12}(\\omega) + P^{(2)}_{21}(\\omega)]$')
    ax[0].plot(frequency_axis, pol_22.real/factor12/4e8, 'navy', label='$\mathcal{Re}[P^{(2)}_{22}(\\omega)]$')
    ax[0].legend(loc=1)
    ax[1].legend(loc=1)

    ax[0].set_ylabel("$\mathcal{Re}[P^{(2)}_{11}(\\omega) + P^{(2)}_{22}(\\omega)]$")
    ax[1].set_ylabel("$\mathcal{Re}[P^{(2)}_{12}(\\omega) + P^{(2)}_{21}(\\omega)]$")
    ax[1].set_xlabel("$400 nm + \Delta \omega$ (in $GHz$)")
    ax[0].yaxis.set_label_position("right")
    ax[1].yaxis.set_label_position("right")

    # for i in range(3):
    #     for label in ax[i].get_xmajorticklabels() + ax[i].get_ymajorticklabels():
    #         label.set_rotation(30)

    # plt.plot(pol_12_21.real, pol_12_21.imag, 'b.')
    # plt.plot(pol_11.real, pol_11.imag, 'r.')
    # plt.plot(pol_22.real, pol_22.imag, 'k.')

    # f, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
    # ax[0].plot([math.atan(z) for z in (pol_12_21.imag/pol_12_21.real)], np.abs(pol_12_21/factor12/4e8), 'r.', label='$P^{(2)}_{12/21}$')
    # ax[0].set_rlabel_position(135)
    # ax[0].legend(loc=6)
    # ax[1].plot([math.atan(z) for z in (pol_11.imag/pol_11.real)], np.abs(pol_11/factor12/4e8), 'b.', label='$P^{(2)}_{11}$')
    # ax[1].plot([math.atan(z) for z in (pol_22.imag/pol_22.real)], np.abs(pol_22/factor12/4e8), 'k.', label='$P^{(2)}_{22}$')
    # ax[1].set_rlabel_position(135)
    # ax[1].legend(loc=6)

    # plt.figure()
    # chi2 = ensemble.calculate_chi(2, 1, **ensemble.molecules[0]) + ensemble.calculate_chi(1, 2, **ensemble.molecules[0])
    # plt.subplot(221)
    # plt.imshow(chi2.real, extent=[frequency_axis.min(), frequency_axis.max(), frequency_axis.min(), frequency_axis.max()])
    # plt.subplot(222)
    # plt.imshow(chi2.imag, extent=[frequency_axis.min(), frequency_axis.max(), frequency_axis.min(), frequency_axis.max()])
    # plt.subplot(223)
    # plt.plot(chi2.sum(axis=1).real)
    # plt.plot(chi2.sum(axis=1).imag)
    # plt.subplot(224)
    # plt.plot(chi2.sum(axis=0).real)
    # plt.plot(chi2.sum(axis=0).imag)
    plt.show()