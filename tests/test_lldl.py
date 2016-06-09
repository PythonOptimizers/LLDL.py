# -*- coding: utf-8 -*-
from lldl.solver import BaseLLDLSolver
import numpy as np
from unittest import TestCase
import pytest
try:
    from cysparse.sparse.ll_mat import *
    import cysparse.common_types.cysparse_types as types
    from lldl.solver import CySparseLLDLSolver
except ImportError:
    pass


class TestLLDLNympy(TestCase):

    def setUp(self):

        # Positive definite matrix
        self.n = 3
        self.indptr = np.array([0, 2, 5, 7])
        self.indices = np.array([0, 1, 0, 1, 2, 1, 2])
        self.values = np.array([2., -1, -1, 2, -1, -1, 2])

        self.colptrT = np.array([0, 1, 2, 2], dtype=np.int64)
        self.rowindT = np.array([1, 2], dtype=np.int64)
        self.valuesT = np.array([-1., - 1.], dtype=np.float64)
        self.adiag = np.array([2.,  2.,  2.], dtype=np.float64)

    def test_init(self):
        lldl = BaseLLDLSolver(self.n, self.adiag, self.colptrT,
                              self.rowindT, self.valuesT, memory=5)
        assert lldl.factored is False

    def test_factorize(self):
        lldl = BaseLLDLSolver(self.n, self.adiag, self.colptrT,
                              self.rowindT, self.valuesT, memory=5)
        lldl.factorize()
        assert lldl.factored is True
        assert np.array_equal(lldl.colptr, np.array([0, 1, 2, 2],
                                                    dtype=np.int64))
        assert np.array_equal(lldl.rowind, np.array([1, 2], dtype=np.int64))
        assert np.allclose(lldl.lvals, np.array([-0.5, -0.666666667]))

    def test_solve(self):
        lldl = BaseLLDLSolver(self.n, self.adiag, self.colptrT,
                              self.rowindT, self.valuesT, memory=5)
        rhs = np.array([1, 0, 1], dtype=np.float64)
        x = lldl.solve(rhs)
        assert np.allclose(x, np.ones(self.n))


class TestLLDLCySparse(TestCase):

    def setUp(self):
        pytest.importorskip("cysparse.common_types.cysparse_types")
        # Positive definite matrix
        self.n = 3
        self.A = LLSparseMatrix(size=3, itype=types.INT64_T,
                                dtype=types.FLOAT64_T)
        self.Anumpy = np.array([[2., -1.,  0.],
                                [-1.,  2., -1.],
                                [0., -1.,  2.]])
        self.A[:, :] = self.Anumpy.copy()

    def test_init(self):
        lldl = CySparseLLDLSolver(self.A, memory=5)
        assert lldl.factored is False

    def test_factorize(self):
        lldl = CySparseLLDLSolver(self.A, memory=5)
        (Ls, d) = lldl.factorize()
        assert lldl.factored is True
        assert PyLLSparseMatrix_Check(Ls) is True

        # strict lower triangular matrix is returned.
        L = Ls + IdentityLLSparseMatrix(size=Ls.nrow)
        Lt = Ls.T + IdentityLLSparseMatrix(size=Ls.nrow)
        D = BandLLSparseMatrix(size=Ls.nrow, diag_coeff=[0], numpy_arrays=[d])
        assert np.allclose((L * D * Lt).to_ll().to_ndarray(), self.Anumpy)

    def test_solve(self):
        lldl = CySparseLLDLSolver(self.A, memory=5)
        rhs = np.array([1, 0, 1], dtype=np.float64)
        x = lldl.solve(rhs)
        assert np.allclose(x, np.ones(self.n))
