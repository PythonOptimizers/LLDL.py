# -*- coding: utf-8 -*-
from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
from lldl.solver import CySparseLLDLSolver
import numpy as np

import sys

# Read Matrix in Matrix Market format from command line argument
A = LLSparseMatrix(mm_filename=sys.argv[1],
                   itype=types.INT64_T,
                   dtype=types.FLOAT64_T)
print "A:"
print A

solver = CySparseLLDLSolver(A, 48)
(Ls, d, a) = solver.factorize()


print 'L:'
print Ls


# strict lower triangular matrix is returned.
L = Ls + IdentityLLSparseMatrix(size=Ls.nrow)
Lt = Ls.T + IdentityLLSparseMatrix(size=Ls.nrow)
D = BandLLSparseMatrix(size=Ls.nrow, diag_coeff=[0], numpy_arrays=[d])

print u"==== LDLᵀ ===="
print (L * D * Lt)

print u"==== Error :  A - LDLᵀ ===="
print ((L * D * Lt) - A).to_ll().to_ndarray()


rhs = A*np.ones(48)
print rhs

x = solver.solve(rhs)

print x

