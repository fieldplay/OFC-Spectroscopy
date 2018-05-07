#include <complex.h>
#include <math.h>
#include <stdio.h>

// Complex type
typedef double complex cmplx;

void pol2_type1(
    cmplx* out,
    const double* freq, const int freq_size,
    const int comb_size, const double delta_freq,
    const double gamma, const double M_field_h, const double M_field_i,
    const double width_g, const cmplx wg_2, const cmplx wg_1, int sign
)
{
    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
    {
        const double omega = freq[out_i];

        cmplx result = 0. + 0. * I;

        for(int p = -comb_size; p < comb_size; p++)
        {
            const cmplx term_A = omega - M_field_h - p * delta_freq - wg_1 + gamma * I;
            const cmplx term_C = M_field_h + p * delta_freq - wg_1 + gamma * I;
            const double c_p = exp(-pow(p,2) / (2.*powf(width_g, 2.)));
            for(int q = -comb_size; q < comb_size; q++)
            {
                const double c_q = exp(-pow(q,2) / (2.*powf(width_g, 2.)));
                const cmplx term_B = M_field_i + q * delta_freq - wg_1 + gamma * I,
                            term_D = omega - M_field_i - q * delta_freq - wg_1 + gamma * I,
                            term_X = M_field_h + M_field_i - omega + (p + q) * delta_freq + 2. * gamma * I,
                            term_X_star = - conj(term_X);
                result += c_p*c_q*(1./(omega - wg_2))*(1./(term_A * term_B) + 1./(term_B * term_X) - 1./(term_A * term_X_star));
//                result += c_p*c_q*(1./(omega - wg_2))*(1./(term_A * term_B) + 1./(term_B * term_X) - 1./(term_A * term_X_star)
//                - 1./(term_C * term_D) - 1./(term_C * term_X) + 1./(term_D * term_X_star));


            }
        }

        out[out_i] += -sign*result;
    }

}

void pol2_total(
    cmplx* out,
    const double* freq, const int freq_size,
    const int comb_size, const double delta_freq,
    const double gamma, const double M_field_h, const double M_field_i, const double M_field_j,
    const double width_g, const cmplx wg_nl, const cmplx wg_ml, const cmplx wg_mn, const cmplx wg_nm
)
{
    pol2_type1(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, width_g, conj(wg_nl), conj(wg_ml), 1);
    pol2_type1(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, width_g, conj(wg_mn), -conj(wg_nl), -1);
    pol2_type1(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, width_g, -wg_nm, conj(wg_ml), -1);
    pol2_type1(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_h, M_field_i, width_g, -wg_ml, -wg_nl, 1);
}