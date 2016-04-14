# -*- coding: utf-8 -*-
from lldl.src.lldl_INT64_FLOAT64 import lldl_INT64_FLOAT64 as sl
from scipy.sparse import csc_matrix, eye, diags, tril
import numpy as np

# Positive definite matrix
n = 3
indptr = np.array([0, 2, 5, 7])
indices = np.array([0, 1, 0, 1, 2, 1, 2])
values = np.array([2., -1, -1, 2, -1, -1, 2])

A = csc_matrix((values, indices, indptr), shape=(n, n))
T = tril(A, -1, format="csc")

print 'A:'
print A
print A.toarray()

adiag = A.diagonal()
colptrT = np.asarray(T.indptr, dtype=np.int64)
rowindT = np.asarray(T.indices, dtype=np.int64)
valuesT = np.asarray(T.data, dtype=np.float64)

(colptr, rowind, lvals, d, alpha) = sl(
    n, n, adiag, colptrT, rowindT, valuesT, memory=5)

L = csc_matrix((lvals, rowind, colptr), shape=(n, n))
print 'L:'
print L.toarray()


L = L + eye(n)  # strict lower triangular matrix is returned.
print u"==== LDLᵀ ===="
print (L * diags(d, 0) * L.T)

print u"==== Error :  A - LDLᵀ ===="
print ((L * diags(d, 0) * L.T) - A).toarray()
