#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef double complex cmplx;

void print_complex_mat(cmplx *A)
{
	int i,j;
	int dim = (sizeof(A)/sizeof(A[0]));
	for(i=0; i<dim; i++)
	{
		for(j=0; j<dim; j++)
		{
			printf("%3.3e + 1j * %3.3e", creal(A[i * dim + j]), cimag(A[i * dim + j]));
		}
	    printf("\n");
	}
	printf("\n\n");
}

void L_operate(cmplx* Qmat, const double field_t, const double* gamma, const cmplx* mu, const double* energies, int nDIM)
//----------------------------------------------------//
// 	    RETURNS Q <-- L[Q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
    int m, n, k;
    complex double* Lmat = (complex double*)malloc(nDIM * nDIM * sizeof(complex double));
    for(m = 0; m < nDIM; m++)
        {
        for(n = 0; n < nDIM; n++)
            {
                Lmat[m * nDIM + n] += - I * (energies[m] - energies[n]) * Qmat[m * nDIM + n];
                for(k = 0; k < nDIM; k++)
                {
                    Lmat[m * nDIM + m] += gamma[k * nDIM + m] * Qmat[k * nDIM + k];
                    Lmat[m * nDIM + n] -= 0.5 * (gamma[n * nDIM + k] + gamma[m * nDIM + k]) * Qmat[m * nDIM + n];
                    Lmat[m * nDIM + n] += I * field_t * (mu[m * nDIM + k] * Qmat[k * nDIM + n] - Qmat[m * nDIM + k] * mu[k * nDIM + n]);
                }

            }

        }

    for(m = 0; m < nDIM; m++)
        {
        for(n = 0; n < nDIM; n++)
            {
                Qmat[m * nDIM + n] = Lmat[m * nDIM + n];
            }
        }

    print_complex_mat(Qmat);

}


void Propagate(cmplx* out, const cmplx* field, const double* gamma, const cmplx* mu, cmplx* rho_0,
     const double* energies, const int timeDIM, const double dt, const int nDIM)
//------------------------------------------------------------//
//    GETTING rho(T) FROM rho(0) USING PROPAGATE FUNCTION     //
//------------------------------------------------------------//
{
    L_operate(rho_0, field[timeDIM/2], gamma, mu, energies, nDIM);
    int m, n;
    for(m = 0; m < nDIM; m++)
        {
        for(n = 0; n < nDIM; n++)
            {
                out[m * nDIM + n] = rho_0[m * nDIM + n];
            }
        }
}