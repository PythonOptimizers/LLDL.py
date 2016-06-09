# -*- coding: utf-8 -*-
from lldl.solver import BaseLLDLSolver
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


lldl = BaseLLDLSolver(n, adiag, colptrT, rowindT, valuesT, memory=5)
lldl.factorize()

L = csc_matrix((lldl.lvals, lldl.rowind, lldl.colptr), shape=(n, n))
print 'L:'
print L.toarray()


L = L + eye(n)  # strict lower triangular matrix is returned.
print u"==== LDLᵀ ===="
print (L * diags(lldl.d, 0) * L.T)

print u"==== Error :  A - LDLᵀ ===="
print ((L * diags(lldl.d, 0) * L.T) - A).toarray()

rhs = A*np.ones([n,1])
print rhs

print lldl.colptr
print lldl.rowind
print lldl.lvals

x = lldl.solve(rhs.flatten())

print x
#
#x = lldl.solve()










