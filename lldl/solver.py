# -*- coding: utf-8 -*-
try:
    from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check, LLSparseMatrix
    from cysparse.common_types.cysparse_types import INT64_T, FLOAT64_T
except ImportError:
    pass
from lldl.src.lldl_INT64_FLOAT64 import lldl_INT64_FLOAT64 as sl
from lldl.src.lldl_INT64_FLOAT64 import dense_lsolve_INT64_FLOAT64
from lldl.src.lldl_INT64_FLOAT64 import dense_ltsolve_INT64_FLOAT64

import numpy as np


class BaseLLDLSolver(object):

    def __init__(self, n, adiag, colptrT, rowindT, valuesT, memory=5):
        """Instantiate a :class:`BaseLLDLSolver` object.

        Perform a limited-memory LDLᵀ factorization of a matrix A

            A = L D Lᵀ
        where L is a lower triangular matrix and D is a diagonal matrix.

        `A` is decomposed as A = T + B + Tᵀ, where T is a strict lower
        triangular matrix supplied in CSC (compressed sparse column) format and
        B is a diagonal matrix supplied as a one dimensional Numpy array.

        The nₖ + p largest elements of L are retained in column k, where nₖ is
        the number of nonzero elements in the strict lower triangle of
        the k-th column of A and where p ∈ N is a limited-memory factor
        specified by the user.

        :warning: matrix `A` must be squared.

        :parameters:
            :n:         number of rows (columns) of A
            :adiag:     diagonal elements of A (as a Numpy array)
            :colptrT:   pointer to column starts in `rowindT` and `valuesT`
                        (as a Nympy array)
            :rowindT:   array of row indices of T (as a Nympy array)
            :valuesT:   array of non zero elements of T (as a Nympy array)
            :memory:    limited-memory factor.
        """
        self.n = n
        self.adiag = adiag.copy()
        self.colptrT = colptrT.copy()
        self.rowindT = rowindT.copy()
        self.valuesT = valuesT.copy()
        self.memory = memory
        self.factored = False
        return

    def factorize(self):
        u"""Factorize matrix A as limited-memory LDLᵀ."""
        (self.colptr, self.rowind, self.lvals, self.d, self.alpha) = \
            sl(self.n, self.n, self.adiag,
               self.colptrT, self.rowindT,
               self.valuesT, memory=self.memory)
        self.factored = True
        return

    def solve(self, rhs):
        u"""Solve A x = b, using the limited-memory LDLᵀ factorization.

        :parameters:
            :rhs:   right-hand side (as a Numpy array).
        """
        if not self.factored:
            self.factorize()

        b = rhs.copy().flatten()
        dense_lsolve_INT64_FLOAT64(self.n, self.colptr,
                                   self.rowind, self.lvals, b)
        z = np.divide(b, self.d)
        dense_ltsolve_INT64_FLOAT64(self.n, self.colptr,
                                    self.rowind, self.lvals, z)
        return z


class CySparseLLDLSolver(BaseLLDLSolver):
    """Specialized class for CySparse matrices."""

    def __init__(self, A, memory=5):
        """Instantiate a :class: `CySparseLLDLSolver` object.

        :parameters:
            :A: the input matrix in CySparse LLmat format.
        """
        if not PyLLSparseMatrix_Check(A):
            raise TypeError('Input matrix should be a `LLSparseMatrix`')

        T = A.to_csc().tril(-1)
        (colptrT, rowindT, valuesT) = T.get_numpy_arrays()
        super(CySparseLLDLSolver, self).__init__(
            A.nrow, A.diag(), colptrT, rowindT, valuesT, memory=memory)

    def factorize(self):
        u"""Factorize matrix A as limited-memory LDLᵀ.

        :returns:
            :L: L as a llmat matrix
            :d: as a Numpy array
        """
        super(CySparseLLDLSolver, self).factorize()
        nnz = len(self.lvals)
        row = np.empty(nnz, dtype=np.int64)
        col = np.empty(nnz, dtype=np.int64)
        val = np.empty(nnz, dtype=np.float64)

        elem = 0
        for j in xrange(len(self.colptr) - 1):
            for k in xrange(self.colptr[j], self.colptr[j + 1]):
                row[elem] = self.rowind[k]
                col[elem] = j
                val[elem] = self.lvals[k]
                elem += 1

        L = LLSparseMatrix(size=self.n,
                           itype=INT64_T,
                           dtype=FLOAT64_T)

        L.put_triplet(row, col, val)
        return (L, self.d)
