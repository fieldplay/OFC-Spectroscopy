import os
import ctypes
from ctypes import c_double, c_int, POINTER, Structure

__doc__ = """
Python wrapper for Pol2Lindblad.c
Compile with:
gcc -O3 -shared -o Pol2Lindblad.so Pol2Lindblad.c -lm -fopenmp -fPIC
"""


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [('real', c_double), ('imag', c_double)]

try:
    # Load the shared library assuming that it is in the same directory
    lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/Pol2Lindblad.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o Pol2Lindblad.so Pol2Lindblad.c -lm -fopenmp -fPIC
        """
    )

#####################################################
#                                                   #
#        Declaring the function Propagate           #
#                                                   #
#####################################################

lib.Propagate.argtypes = (
    POINTER(c_complex),     # cmplx* out, , Array to store L[Q]
    POINTER(c_complex),     # cmplx* field, E(t)
    POINTER(c_double),      # double* gamma matrix
    POINTER(c_complex),      # double* mu matrix
    POINTER(c_complex),      # double* rho_0 matrix
    POINTER(c_double),      # double* energies
    c_int,                  # const int timeDIM,
    c_double,               # const double dt,
    c_int                   # const int nDIM
)
lib.Propagate.restype = None


def Propagate(out, field, gamma, mu, rho_0, energies, timeDIM, dt):
    return lib.Propagate(
        out.ctypes.data_as(POINTER(c_complex)),
        field.ctypes.data_as(POINTER(c_complex)),
        gamma.ctypes.data_as(POINTER(c_double)),
        mu.ctypes.data_as(POINTER(c_complex)),
        rho_0.ctypes.data_as(POINTER(c_complex)),
        energies.ctypes.data_as(POINTER(c_double)),
        timeDIM,
        dt,
        len(energies)
    )