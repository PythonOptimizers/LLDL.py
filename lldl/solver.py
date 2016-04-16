# -*- coding: utf-8 -*-
from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check, LLSparseMatrix
from cysparse.common_types.cysparse_types import INT64_T, FLOAT64_T
from lldl.src.lldl_INT64_FLOAT64 import lldl_INT64_FLOAT64 as sl
import numpy as np


class BaseLLDLSolver(object):
    """."""

    def __init__(self, n, adiag, colptrT, rowindT, valuesT, memory=5):
        self.adiag = adiag.copy()
        self.colptrT = colptrT.copy()
        self.rowindT = rowindT.copy()
        self.valuesT = valuesT.copy()
        self.memory = memory
        return

    def factorize():
        pass

    def solve():
        pass


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
        print T
        (colptrT, rowindT, valuesT) = T.get_numpy_arrays()
        print "colptrT:", colptrT, colptrT.dtype
        print "rowindT:", rowindT
        print "valuesT:", valuesT
        self.n = A.nrow
        super(CySparseLLDLSolver, self).__init__(
            self.n, A.diag(), colptrT, rowindT, valuesT, memory=memory)

    def factorize(self):
        """."""
        print "python:", self.colptrT
        (colptr, rowind, lvals, d, alpha) = sl(self.n, self.n, self.adiag,
                                               self.colptrT, self.rowindT, self.valuesT,
                                               memory=self.memory)
        print "done"
        print lvals
        nnz = len(lvals)
        row = np.empty(nnz, dtype=np.int64)
        col = np.empty(nnz, dtype=np.int64)
        val = np.empty(nnz, dtype=np.float64)

        elem = 0
        for j in xrange(len(colptr) - 1):
            for k in xrange(colptr[j], colptr[j + 1]):
                row[elem] = rowind[k]
                col[elem] = j
                val[elem] = lvals[k]
                elem += 1

        L = LLSparseMatrix(size=self.n,
                           itype=INT64_T,
                           dtype=FLOAT64_T)

        L.put_triplet(row, col, val)
        return (L, d, alpha)
