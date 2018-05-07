__doc__ = """
Python wrapper for eval_pol3.c

Note: You must compile the C shared library
       gcc -O3 -shared -o eval_pol3.so eval_pol3.c -lm -fopenmp
"""
import os
import ctypes
from ctypes import c_double, c_int, POINTER, Structure


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [('real', c_double), ('imag', c_double)]

try:
    # Load the shared library assuming that it is in the same directory
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/eval_pol3.so")
except OSError:
    raise NotImplementedError(
        """The library is absent. You must compile the C shared library using the commands:
              gcc -O3 -shared -o eval_pol3.so eval_pol3.c -lm -fopenmp
        """
    )

############################################################################################
#
#   Declaring the function pol3_a2
#
############################################################################################

lib.pol3_total.argtypes = (
    POINTER(c_complex), # cmplx* out, # Array to save the polarizability
    POINTER(c_double),  # double* freq, frequency arrays
    c_int,      # const int freq_size,
    c_int,      # const int comb_size,
    c_double,   # const double delta_freq,
    c_double,   # const double gamma,
    c_double,   # const double M_field1,
    c_double,   # const double M_field2,
    c_double,   # const double M_field3, // Comb parameters
    c_double,   # const double width_g
    c_complex,  # const cmplx wg_nv,
    c_complex,  # const cmplx wg_mv,
    c_complex,  # const cmplx wg_vl,
    c_complex,  # const cmplx wg_nl,
    c_complex,  # const cmplx wg_ml,
    c_complex,  # const cmplx wg_mn,
    c_complex,  # const cmplx wg_nm,
    c_complex,  # const cmplx wg_vn,
    c_complex   # const cmplx wg_vm // omega_{ij} + I * gamma_{ij} for each transition from i to j
)
lib.pol3_total.restype = None


def pol3_total(out, params, M_field_h, M_field_i, M_field_j, wg_nv, wg_mv, wg_vl, wg_nl, wg_ml, wg_mn, wg_nm, wg_vn, wg_vm):
    return lib.pol3_total(
        out.ctypes.data_as(POINTER(c_complex)),
        params.freq.ctypes.data_as(POINTER(c_double)),
        out.size,
        params.comb_size,
        params.delta_freq,
        params.gamma,
        M_field_h,
        M_field_i,
        M_field_j,
        params.width_g,
        c_complex(wg_nv.real, wg_nv.imag),
        c_complex(wg_mv.real, wg_mv.imag),
        c_complex(wg_vl.real, wg_vl.imag),
        c_complex(wg_nl.real, wg_nl.imag),
        c_complex(wg_ml.real, wg_ml.imag),
        c_complex(wg_mn.real, wg_mn.imag),
        c_complex(wg_nm.real, wg_nm.imag),
        c_complex(wg_vn.real, wg_vn.imag),
        c_complex(wg_vm.real, wg_vm.imag),
    )