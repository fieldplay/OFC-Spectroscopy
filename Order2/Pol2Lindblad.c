#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef double complex cmplx;

void print_complex_mat(cmplx *A)
{
	int i,j;
	int nDIM = 3;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e + %3.3eJ  ", creal(A[i * nDIM + j]), cimag(A[i * nDIM + j]));
		}
	    printf("\n");
	}
	printf("\n\n");
}

void copy_mat(cmplx *A, cmplx *B)
//----------------------------------------------------//
// 	        COPIES MATRIX A ----> MATRIX B            //
//----------------------------------------------------//
{
    int i, j = 0;
    int nDIM = 3;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] = A[i * nDIM + j];
        }
    }
}

void add_mat(cmplx *A, cmplx *B)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    int nDIM = 3;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] += A[i * nDIM + j];
        }
    }
}

void scale_mat(cmplx *A, double factor)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    int nDIM = 3;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            A[i * nDIM + j] *= factor;
        }
    }
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
                    if (m == n)
                    {
                        Lmat[m * nDIM + m] += gamma[k * nDIM + m] * Qmat[k * nDIM + k];
                    }
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
    int i, j, m, n;

    complex double* L_func = (complex double*)malloc(nDIM * nDIM * sizeof(complex double));
    copy_mat(rho_0, L_func);
    copy_mat(rho_0, out);

    for(i=0; i<100; i++)
    {
        for(j=0; j<10; j++)
        {
            L_operate(L_func, field[timeDIM/2], gamma, mu, energies, nDIM);
            scale_mat(L_func, dt/(j+1));
            add_mat(L_func, out);
        }
    }
}