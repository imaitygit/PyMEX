cdef extern from "complex.h" nogil:
  double complex cexp(double complex)
  double complex conj(double complex)
  double complex cabs(double complex)
