#include <complex.h>
#include <math.h>
#include <stdio.h>

// Complex type
typedef double complex cmplx;

void pol3(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const int comb_size, const double delta_freq,
    const double gamma, const double M_field_h, const double M_field_i, const double M_field_j, // Comb parameters
    const double width_g, const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign // omega_{ij} + I * gamma_{ij} for each transition from i to j
)
{
    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
    {
        const double omega = freq[out_i];

        cmplx result = 0. + 0. * I;

        for(int p = -comb_size; p < comb_size; p++)
        {
            const cmplx term_R = M_field_h + p * delta_freq - wg_1 + gamma * I;
            const double c_p = exp(-pow(p,2) / (2.*powf(width_g, 2.)));
            for(int q = -comb_size; q < comb_size; q++)
            {
                const cmplx term_Y = M_field_h + M_field_i + (p + q) * delta_freq - wg_2 + 2. * gamma * I;
                const double c_q = exp(-pow(q,2) / (2.*powf(width_g, 2.)));
                for(int r = -comb_size; r < comb_size; r++)
                {
                    const double c_r = exp(-pow(r,2) / (2.*powf(width_g, 2.)));
                    const cmplx term_Z = - M_field_h - M_field_i + M_field_j + (p + q - r) * delta_freq + omega + 3. * gamma * I,
                                term_Z_star = conj(term_Z),
                                term_S = omega - M_field_i + M_field_j + (r - q) * delta_freq - wg_1 + 2. * gamma * I,
                                term_X = omega + M_field_j + r * delta_freq - wg_2 + gamma * I;
                    result += (1./(wg_3 - omega))*(1./(term_X * term_Y * term_R) + 1./(term_X * term_S * term_R)
                                          + 1./(term_X * term_S * term_Z) - 1./(term_Y * term_R * term_Z_star));
                }

            }
        }

        out[out_i] += sign*result*3.*M_PI*M_PI;
    }

}

void pol3_total(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const int comb_size, const double delta_freq,
    const double gamma, const double M_field_h, const double M_field_i, const double M_field_j, // Comb parameters
    const double width_g, const cmplx wg_nv, const cmplx wg_mv, const cmplx wg_vl, const cmplx wg_nl, const cmplx wg_ml, const cmplx wg_mn,
    const cmplx wg_nm, const cmplx wg_vn, const cmplx wg_vm // omega_{ij} + I * gamma_{ij} for each transition from i to j
)
{
    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, conj(wg_vl), conj(wg_nl), -conj(wg_vl), -1);
    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, conj(wg_nv), conj(wg_mv), wg_vl, 1);
    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, conj(wg_nv), -wg_vm, -conj(wg_ml), 1);
    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, conj(wg_mn), -wg_nl, wg_vl, -1);
    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, -wg_vn, conj(wg_nl), -conj(wg_ml), 1);
    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, -wg_nm, conj(wg_mv), wg_vl, -1);
    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, -wg_nm, -wg_mv, -conj(wg_ml), -1);
    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, -wg_ml, -wg_nl, wg_vl, 1);
}