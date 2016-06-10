cimport numpy as cnp

cdef extern from "math.h":
    double sqrt(double x) nogil


cdef bint attempt_lldl_INT64_FLOAT64(unsigned int n,
                                      cnp.ndarray[cnp.float64_t, ndim=1] d,
                                      cnp.ndarray[cnp.float64_t, ndim=1] lvals,
                                      cnp.ndarray[cnp.int64_t, ndim=1] rowind,
                                      cnp.ndarray[cnp.int64_t, ndim=1] colptr,
                                      unsigned int memory=?)

cdef cnp.int64_t flip_INT64(cnp.int64_t i)

cdef cnp.int64_t unflip_INT64(cnp.int64_t i)

cdef bint marked_INT64(cnp.int64_t * w, cnp.int64_t j)

cdef mark_INT64(cnp.int64_t * w, cnp.int64_t j)

cdef reach_INT64(cnp.int64_t n,
                    cnp.int64_t * Gi,
                    cnp.int64_t * Gr,
                    cnp.int64_t * Bi,
                    cnp.int64_t * Br,
                    int k,
                    cnp.ndarray[cnp.int64_t, ndim=1] xi)

cdef cnp.int64_t deep_first_sort_INT64_FLOAT64(int j,
                                                          cnp.int64_t * Gi,
                                                          cnp.int64_t * Gr,
                                                          cnp.int64_t top,
                                                          cnp.int64_t * xi,
                                                          cnp.int64_t * pstack,
                                                          const cnp.int64_t * pinv)