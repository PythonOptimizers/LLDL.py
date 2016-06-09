# -*- coding: utf-8 -*-
from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check, LLSparseMatrix
from cysparse.common_types.cysparse_types import INT64_T, FLOAT64_T
from lldl.src.lldl_INT64_FLOAT64 import lldl_INT64_FLOAT64 as sl
from lldl.src.lldl_INT64_FLOAT64 import dense_lsolve_INT64_FLOAT64
from lldl.src.lldl_INT64_FLOAT64 import dense_ltsolve_INT64_FLOAT64

import numpy as np


class BaseLLDLSolver(object):
    """."""

    def __init__(self, n, adiag, colptrT, rowindT, valuesT, memory=5):
        self.n = n
        self.adiag = adiag.copy()
        self.colptrT = colptrT.copy()
        self.rowindT = rowindT.copy()
        self.valuesT = valuesT.copy()
        self.memory = memory
        return

    def factorize(self):
        (self.colptr, self.rowind, self.lvals, self.d, self.alpha) = sl(self.n, self.n, self.adiag,
                                                                        self.colptrT, self.rowindT,
                                                                        self.valuesT, memory=self.memory)
        return

    def solve(self, rhs):
        b = rhs.copy().flatten()
        print self.n
        dense_lsolve_INT64_FLOAT64(self.n, self.colptr,
                                   self.rowind, self.lvals, b)
        z = np.divide(b, self.d)
        dense_ltsolve_INT64_FLOAT64(self.n, self.colptr,
                                    self.rowind, self.lvals, z)
        return z


class CySparseLLDLSolver(BaseLLDLSolver):
    """Specialized class for CySparse matrices."""

    def __init__(self, A, memory=5):
        """Instantiate a :class: `CySparseLLDLSolver`.

        :parameters:
            :A: the input matrix
        """
        if not PyLLSparseMatrix_Check(A):
            raise TypeError('Input matrix should be a `LLSparseMatrix`')

        T = A.to_csc().tril(-1)
        (colptrT, rowindT, valuesT) = T.get_numpy_arrays()
        super(CySparseLLDLSolver, self).__init__(
            A.nrow, A.diag(), colptrT, rowindT, valuesT, memory=memory)

    def factorize(self):
        """."""
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
        return (L, self.d, self.alpha)
