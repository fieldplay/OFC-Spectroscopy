#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef double complex cmplx;

void print_complex_mat(cmplx *A, int nDIM)
{
	int i,j;
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

void print_double_mat(double *A, int nDIM)
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e  ", A[i * nDIM + j]);
		}
	    printf("\n");
	}
	printf("\n\n");
}

void copy_mat(const cmplx *A, cmplx *B, int nDIM)
//----------------------------------------------------//
// 	        COPIES MATRIX A ----> MATRIX B            //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] = A[i * nDIM + j];
        }
    }
}

void add_mat(const cmplx *A, cmplx *B, int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] += A[i * nDIM + j];
        }
    }
}

void scale_mat(cmplx *A, double factor, int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            A[i * nDIM + j] *= factor;
        }
    }
}

double complex_abs(cmplx z)
{
    return sqrt((creal(z)*creal(z) + cimag(z)*cimag(z)));
}

cmplx complex_trace(cmplx *A, int nDIM)
{
    cmplx trace = 0.0 + I * 0.0;
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        trace += A[i*nDIM + i];
    }
    printf("Trace = %3.3e + %3.3eJ  \n", creal(trace), cimag(trace));

    return trace;
}

cmplx *multiply_complex_mat(cmplx *A, cmplx *B, int nDIM)
{
    cmplx* product = (cmplx*)calloc(nDIM * nDIM,  sizeof(cmplx));
    int i, j, k;
    for (i=0; i<nDIM; i++)
    {
        for (j=0; j<nDIM; j++)
        {
            for (k=0; k<nDIM; k++)
            {
                product[i*nDIM + j] += A[i*nDIM + k]*B[k*nDIM + j];
            }
        }
    }

    return product;
}
double complex_max_element(cmplx *A, int nDIM)
{
    double max_el = 0.0;
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            if(complex_abs(A[i * nDIM + j]) > max_el)
            {
                max_el = complex_abs(A[i * nDIM + j]);
            }
        }
    }
    return max_el;
}

void L_operate(cmplx* Qmat, const cmplx field_t, const double* gamma, const cmplx* mu, const double* energies, int nDIM)
//----------------------------------------------------//
// 	    RETURNS Q <-- L[Q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
    int m, n, k;
    cmplx* Lmat = (cmplx*)calloc(nDIM * nDIM,  sizeof(cmplx));
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


}


void Propagate(cmplx* out, cmplx* dyn_rho, cmplx* dyn_coh, const cmplx* field, const double* gamma, cmplx* mu, const cmplx* rho_0,
     const double* energies, const int timeDIM, const double dt, const int nDIM)
//------------------------------------------------------------//
//    GETTING rho(T) FROM rho(0) USING PROPAGATE FUNCTION     //
//------------------------------------------------------------//
{
    int i, j, k;
    cmplx* L_func = (cmplx*)calloc(nDIM * nDIM, sizeof(cmplx));
    copy_mat(rho_0, L_func, nDIM);
    copy_mat(rho_0, out, nDIM);
    for(i=0; i<timeDIM; i++)
    {
        j=0;
        do
        {
            L_operate(L_func, field[i], gamma, mu, energies, nDIM);
            scale_mat(L_func, dt/(j+1), nDIM);
            add_mat(L_func, out, nDIM);
            j+=1;
        }while(complex_max_element(L_func, nDIM) > 1.0E-12);

        for(k=0; k<nDIM; k++)
        {
            dyn_rho[k*timeDIM + i] = out[k*nDIM + k];
        }

        dyn_coh[i] = out[1];

        copy_mat(out, L_func, nDIM);
//        print_complex_mat(out);
//        pol2[i] = out[1] + out[2] + out[3] + out[5] + out[6] + out[7];
    }
}
