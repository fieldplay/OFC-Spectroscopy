#include <complex.h>
#include <math.h>
#include <stdio.h>

// Complex type
typedef double complex cmplx;

void pol3_XYR(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const double delta_freq, const double gamma, const double M_field_h, const double M_field_i, const double M_field_j,
    const double width_g, int N_terms, const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign
)
{
    int p0 = ceil((crealf(wg_1) - M_field_h)/delta_freq);
    int q0 = ceil((crealf(wg_2) - M_field_h - M_field_i)/delta_freq) - p0;

    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
        {
            const double omega = freq[out_i];

            cmplx result = 0. + 0. * I;
            int r0 = ceil((crealf(wg_2) - omega - M_field_j)/delta_freq);

            for(int p = p0 - N_terms; p < p0 + N_terms; p++)
            {
                const cmplx term_R = M_field_h + p * delta_freq - wg_1 + gamma * I;
                for(int q = q0 - N_terms; q < q0 + N_terms; q++)
                {
                    const cmplx term_Y = M_field_h + M_field_i + (p + q) * delta_freq - wg_2 + 2. * gamma * I;
                    for(int r = r0 - N_terms; r < r0 + N_terms; r++)
                    {
                        const cmplx term_X = omega + M_field_j + r * delta_freq - wg_2 + gamma * I;
                        result += 1./(term_X * term_Y * term_R);
                    }

                }
            }

            out[out_i] += sign*result/(omega - wg_3);
        }

}

void pol3_XSR(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const double delta_freq, const double gamma, const double M_field_h, const double M_field_i, const double M_field_j,
    const double width_g, int N_terms, const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign
)
{
    int p0 = ceil((crealf(wg_1) - M_field_h)/delta_freq);

    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
        {
            const double omega = freq[out_i];

            cmplx result = 0. + 0. * I;
            int r0 = ceil((crealf(wg_2) - omega - M_field_j)/delta_freq);
            int q0 = r0 - ceil((crealf(wg_2) - M_field_h - M_field_i)/delta_freq);

            for(int p = p0 - N_terms; p < p0 + N_terms; p++)
            {
                const cmplx term_R = M_field_h + p * delta_freq - wg_1 + gamma * I;
                for(int q = q0 - N_terms; q < q0 + N_terms; q++)
                {
                    for(int r = r0 - N_terms; r < r0 + N_terms; r++)
                    {
                        const cmplx term_X = omega + M_field_j + r * delta_freq - wg_2 + gamma * I;
                        const cmplx term_S = omega + M_field_j - M_field_i + (r - q) * delta_freq - wg_1 + 2. * gamma * I;
                        result += 1./(term_X * term_S * term_R);
                    }

                }
            }

            out[out_i] += sign*result/(omega - wg_3);
        }

}

void pol3_XSZ(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const double delta_freq, const double gamma, const double M_field_h, const double M_field_i, const double M_field_j,
    const double width_g, int N_terms, const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign
)
{
    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
        {
            const double omega = freq[out_i];

            cmplx result = 0. + 0. * I;
            int r0 = ceil((crealf(wg_2) - omega - M_field_j)/delta_freq);
            int q0 = r0 - ceil((crealf(wg_1) - omega - M_field_j + M_field_i)/delta_freq);
            int p0 = ceil((omega - (M_field_h + M_field_i - M_field_j))/delta_freq)
                            + ceil((crealf(wg_1) - omega - M_field_j + M_field_i)/delta_freq);

            for(int p = p0 - N_terms; p < p0 + N_terms; p++)
            {
                for(int q = q0 - N_terms; q < q0 + N_terms; q++)
                {
                    for(int r = r0 - N_terms; r < r0 + N_terms; r++)
                    {
                        const cmplx term_X = omega + M_field_j + r * delta_freq - wg_2 + gamma * I;
                        const cmplx term_S = omega + M_field_j - M_field_i + (r - q) * delta_freq - wg_1 + 2. * gamma * I;
                        const cmplx term_Z = omega - (M_field_h + M_field_i - M_field_j) - (p + q - r) * delta_freq + 3. * gamma * I;
                        result += 1./(term_X * term_S * term_Z);
                    }

                }
            }

            out[out_i] += sign*result/(omega - wg_3);
        }

}

void pol3_YRZstar(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const double delta_freq, const double gamma, const double M_field_h, const double M_field_i, const double M_field_j,
    const double width_g, int N_terms, const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign
)
{
    int p0 = ceil((crealf(wg_1) - M_field_h)/delta_freq);
    int q0 = ceil((crealf(wg_2) - M_field_h - M_field_i)/delta_freq) - p0;

    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
        {
            const double omega = freq[out_i];

            cmplx result = 0. + 0. * I;
            int r0 = p0 + q0 - ceil((omega - (M_field_h + M_field_i - M_field_j))/delta_freq);

            for(int p = p0 - N_terms; p < p0 + N_terms; p++)
            {
                const cmplx term_R = M_field_h + p * delta_freq - wg_1 + gamma * I;
                for(int q = q0 - N_terms; q < q0 + N_terms; q++)
                {
                    const cmplx term_Y = M_field_h + M_field_i + (p + q) * delta_freq - wg_2 + 2. * gamma * I;
                    for(int r = r0 - N_terms; r < r0 + N_terms; r++)
                    {
                        const cmplx term_Zstar = omega - (M_field_h + M_field_i - M_field_j) - (p + q - r) * delta_freq - 3. * gamma * I;
                        result += 1./(term_Y * term_R * term_Zstar);
                    }

                }
            }

            out[out_i] -= sign*result/(omega - wg_3);
        }

}

void pol3(
    cmplx* out,
    const double* freq, const int freq_size,
    const double delta_freq, const double gamma, const double M_field_h, const double M_field_i, const double M_field_j,
    const double width_g, const int N_terms, const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, int sign
)
{
    pol3_XYR(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, wg_3, wg_2, wg_1, sign);
    pol3_XSR(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, wg_3, wg_2, wg_1, sign);
    pol3_XSZ(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, wg_3, wg_2, wg_1, sign);
    pol3_YRZstar(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, wg_3, wg_2, wg_1, sign);

}



void pol3_total(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const double delta_freq, const double gamma, const double M_field_h, const double M_field_i, const double M_field_j,
    const double width_g, const int N_terms, const cmplx wg_nv, const cmplx wg_mv, const cmplx wg_vl, const cmplx wg_nl,
    const cmplx wg_ml, const cmplx wg_mn, const cmplx wg_nm, const cmplx wg_vn, const cmplx wg_vm
)
{
    pol3(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, wg_vl, wg_nl, wg_ml, -1);
    pol3(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, wg_nv, wg_mv, -conj(wg_vl), 1);
    pol3(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, wg_nv, -conj(wg_vm), wg_ml, 1);
    pol3(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, wg_mn, -conj(wg_nl), -conj(wg_vl), -1);
    pol3(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, -conj(wg_vn), wg_nl, wg_ml, 1);
    pol3(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, -conj(wg_nm), wg_mv, -conj(wg_vl), -1);
    pol3(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, -conj(wg_nm), -conj(wg_mv), wg_ml, -1);
    pol3(out, freq, freq_size, delta_freq, gamma, M_field_h, M_field_i, M_field_j, width_g, N_terms, -conj(wg_ml), -conj(wg_nl), -conj(wg_vl), 1);

}