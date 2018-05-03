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

        const cmplx term_U = wg_3 - omega;

        for(int h = -comb_size; h < comb_size; h++)
        {
            const cmplx term_V = M_field_h + h * delta_freq + wg_1 + gamma * I;
            const double c_h = exp(-pow(h,2) / (2.*powf(width_g, 2.)));
            for(int i = -comb_size; i < comb_size; i++)
            {
                const cmplx term_X = M_field_h + M_field_i + (h + i) * delta_freq - wg_2 + 2. * gamma * I;
                const double c_i = exp(-pow(i,2) / (2.*powf(width_g, 2.)));
                for(int j = -comb_size; j < comb_size; j++)
                {
                    const double c_j = exp(-pow(j,2) / (2.*powf(width_g, 2.)));
                    const cmplx term_Y = M_field_h + M_field_i + M_field_j + (h + i + j) * delta_freq + omega + 3. * gamma * I,
                                term_Y_star = - conj(term_Y),
                                term_W = omega - M_field_i - M_field_j - (i + j) * delta_freq + wg_1 + 2. * gamma * I,
                                term_Z = omega - M_field_j - j * delta_freq - wg_2 + gamma * I;
                    result += c_h*c_i*c_j*(1./term_U)*(1./(term_Z * term_V * term_X) + 1./(term_V * term_X * term_Y)
                                          + 1./(term_Z * term_V * term_W) + 1./(term_Z * term_W * term_Y_star));
                }

            }
        }

        out[out_i] += sign*result;
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
//    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, conj(wg_nv), conj(wg_mv), wg_vl, 1);
//    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, conj(wg_nv), -wg_vm, -conj(wg_ml), 1);
//    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, conj(wg_mn), -wg_nl, wg_vl, -1);
//    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, -wg_vn, conj(wg_nl), -conj(wg_ml), 1);
//    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, -wg_nm, conj(wg_mv), wg_vl, -1);
//    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, -wg_nm, -wg_mv, -conj(wg_ml), -1);
//    pol3(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, -wg_ml, -wg_nl, wg_vl, 1);
}