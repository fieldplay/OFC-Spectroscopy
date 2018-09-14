#include <complex.h>
#include <math.h>
#include <stdio.h>

// Complex type
typedef double complex cmplx;

void pol2_XY(
    cmplx* out, const double* freq, const int freq_size, const double delta_freq, const double gamma,
    const double M_field_p, const double M_field_q, const double width_g, int N_terms,
    const cmplx wg_2, const cmplx wg_1, int sign
)
{
    int m_p0 = (crealf(wg_1) - M_field_p)/delta_freq;

    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
        {
            const double omega = freq[out_i];
            int m_q0 = omega - M_field_q - (crealf(wg_1))/delta_freq;

            cmplx result = 0. + 0. * I;

            for(int m_p = m_p0 - N_terms; m_p < m_p0 + N_terms; m_p++)
            {
                const cmplx term_X = M_field_p + m_p * delta_freq - wg_1 + gamma * I;
                for(int m_q = m_q0 - N_terms; m_q < m_q0 + N_terms; m_q++)
                {
                    const cmplx term_Y = omega - (M_field_q + m_q * delta_freq - gamma * I) - wg_1 ;
                    result += 1./(term_X * term_Y);
                }
            }

            out[out_i] += result*sign*I/(omega - wg_2);
        }

}

void pol2_XZ(
    cmplx* out, const double* freq, const int freq_size, const double delta_freq, const double gamma,
    const double M_field_p, const double M_field_q, const double width_g, int N_terms,
    const cmplx wg_2, const cmplx wg_1, int sign
)
{
    int m_p0 = (crealf(wg_1) - M_field_p)/delta_freq;

    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
        {
            const double omega = freq[out_i];

            cmplx result = 0. + 0. * I;
            int m_q0 = (omega - M_field_q - M_field_p)/delta_freq - m_p0;

            for(int m_p = m_p0 - N_terms; m_p < m_p0 + N_terms; m_p++)
            {
                const cmplx term_X = M_field_p + m_p * delta_freq - wg_1 + gamma * I;
                for(int m_q = m_q0 - N_terms; m_q < m_q0 + N_terms; m_q++)
                {
                    const cmplx term_Z = M_field_p + M_field_q - omega + (m_p + m_q) * delta_freq + 2 * I * gamma;
                    result += 1./(term_X * term_Z);
                }
            }

            out[out_i] += result*sign*I/(omega - wg_2);
        }

}

void pol2_YZstar(
    cmplx* out, const double* freq, const int freq_size, const double delta_freq, const double gamma,
    const double M_field_p, const double M_field_q, const double width_g, int N_terms,
    const cmplx wg_2, const cmplx wg_1, int sign
)
{
    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
        {
            const double omega = freq[out_i];
            int m_q0 = (omega - M_field_q - crealf(wg_1))/delta_freq;
            int m_p0 = (omega - M_field_q - M_field_p)/delta_freq - m_q0;

            cmplx result = 0. + 0. * I;

            for(int m_q = m_q0 - N_terms; m_q < m_q0 + N_terms; m_q++)
            {
                for(int m_p = m_p0 - N_terms; m_p < m_p0 + N_terms; m_p++)
                {
                    const cmplx term_Zstar = omega - (M_field_p + M_field_q  + (m_p + m_q) * delta_freq) + 2 * I * gamma;
                    const cmplx term_X = M_field_p + m_p * delta_freq - wg_1 + gamma * I;
                    result -= 1./(term_X * term_Zstar);
                }
            }

            out[out_i] += result*sign*I/(omega - wg_2);
        }

}

void pol2(
    cmplx* out, const double* freq, const int freq_size, const double delta_freq, const double gamma,
    const double M_field_p, const double M_field_q, const double width_g, const int N_terms,
    const cmplx wg_2, const cmplx wg_1, int sign
)
{
    pol2_XY(out, freq, freq_size, delta_freq, gamma, M_field_p, M_field_q, width_g, N_terms, wg_2, wg_1, sign);
    pol2_XZ(out, freq, freq_size, delta_freq, gamma, M_field_p, M_field_q, width_g, N_terms, wg_2, wg_1, sign);
    pol2_YZstar(out, freq, freq_size, delta_freq, gamma, M_field_p, M_field_q, width_g, N_terms, wg_2, wg_1, sign);
}



void pol2_total(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const double delta_freq, const double gamma, const double M_field_p, const double M_field_q,
    const double width_g, const int N_terms, const cmplx wg_nl, const cmplx wg_ml, const cmplx wg_mn, const cmplx wg_nm
)
{
    pol2(out, freq, freq_size, delta_freq, gamma, M_field_p, M_field_q, width_g, N_terms, conj(wg_nl), conj(wg_ml), 1);
    pol2(out, freq, freq_size, delta_freq, gamma, M_field_p, M_field_q, width_g, N_terms, conj(wg_mn), -wg_nl, -1);
    pol2(out, freq, freq_size, delta_freq, gamma, M_field_p, M_field_q, width_g, N_terms, -wg_nm, conj(wg_ml), -1);
    pol2(out, freq, freq_size, delta_freq, gamma, M_field_p, M_field_q, width_g, N_terms, -wg_ml, -wg_nl, 1);
}