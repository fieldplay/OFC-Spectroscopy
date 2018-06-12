__doc__ = """
Python wrapper for eval_pol2_full.c

Note: You must compile the C shared library
       gcc -O3 -shared -o eval_pol2_full.so eval_pol2_full.c -lm -fopenmp
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
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/eval_pol2_full.so")
except OSError:
    raise NotImplementedError(
        """The library is absent. You must compile the C shared library using the commands:
              gcc -O3 -shared -o eval_pol2_full.so eval_pol2_full.c -lm -fopenmp
        """
    )

############################################################################################
#
#   Declaring the function pol2_a2
#
############################################################################################

lib.pol2_total.argtypes = (
    POINTER(c_complex), # cmplx* out, # Array to save the polarizability
    POINTER(c_double),  # double* freq, frequency arrays
    c_int,      # const int freq_size,
    c_double,   # const double delta_freq,
    c_double,   # const double gamma,
    c_double,   # const double M_field_p,
    c_double,   # const double M_field_q,
    c_double,   # const double width_g
    c_int,      # const int N_terms
    c_complex,  # const cmplx wg_nl,
    c_complex,  # const cmplx wg_ml,
    c_complex,  # const cmplx wg_mn,
    c_complex,  # const cmplx wg_nm,
)
lib.pol2_total.restype = None


def pol2_total(out, params, M_field_p, M_field_q, wg_nl, wg_ml, wg_mn, wg_nm):
    return lib.pol2_total(
        out.ctypes.data_as(POINTER(c_complex)),
        params.freq.ctypes.data_as(POINTER(c_double)),
        out.size,
        params.delta_freq,
        params.gamma,
        M_field_p,
        M_field_q,
        params.width_g,
        params.N_terms,
        c_complex(wg_nl.real, wg_nl.imag),
        c_complex(wg_ml.real, wg_ml.imag),
        c_complex(wg_mn.real, wg_mn.imag),
        c_complex(wg_nm.real, wg_nm.imag),
    )