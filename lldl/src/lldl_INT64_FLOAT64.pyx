# -*- coding: utf-8 -*-
"""Base class for the LLDL factorization."""
cimport numpy as cnp
import numpy as np

import sys

cdef extern from "math.h":
    double sqrt(double x) nogil

cimport cython


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef lldl_INT64_FLOAT64(unsigned int n, unsigned int m,
                          cnp.ndarray[cnp.float64_t, ndim=1] adiag,
                          cnp.ndarray[cnp.int64_t, ndim=1] colptrT,
                          cnp.ndarray[cnp.int64_t, ndim=1] rowindT,
                          cnp.ndarray[cnp.float64_t, ndim=1] valuesT,
                          unsigned int memory):
    """"Compute the incomplete LDL^T factorization of A with limited-memory.

    parameter `memory` and shift alpha.
    """
    if n != m:
        raise TypeError("Input matrix must be square")

    if memory < 0:
        raise ValueError("Limited-memory parameter must be nonnegative")

    cdef cnp.ndarray[cnp.float64_t, ndim = 1] wa1 = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim = 1] s = np.ones(n, dtype=np.float64)

    cdef unsigned int col, k, row
    cdef unsigned int max_increase_alpha, nb_increase_alpha
    cdef unsigned int nnzT = valuesT.shape[0]
    cdef unsigned int n_mem = n * memory

    cdef bint success, tired
    cdef cnp.float64_t alpha = 0.0
    cdef cnp.float64_t alpha_min, val

    # Make room to store L
    cdef unsigned int nnzLmax = max(nnzT + n_mem, np.divide(n * (n - 1), 2))
    cdef cnp.ndarray[cnp.float64_t, ndim = 1] d = np.zeros(n)
    cdef cnp.ndarray[cnp.float64_t, ndim = 1] lvals = np.zeros(nnzLmax)
    cdef cnp.ndarray[cnp.int64_t, ndim = 1] rowind = np.zeros(nnzLmax, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim = 1] colptr = np.zeros(n + 1, dtype=np.int64)

    # Compute the 2-norm of columns of A.
    for col in xrange(n):
        for k in xrange(colptrT[col], colptrT[col + 1]):
            row = rowindT[k]
            val = valuesT[k]
            wa1[col] += val * val  # Contribution to column col.
            wa1[row] += val * val  # Contribution to column row.
    for col in xrange(n):
        wa1[col] += adiag[col] * adiag[col]
        wa1[col] = sqrt(wa1[col])

    # Compute scaling matrix.
    for col in xrange(n):
        if wa1[col] > 0:
            s[col] = 1 / sqrt(wa1[col])

    # Set initial shift. Keep it at zero if possible.
    alpha_min = 1.0e-3
    max_increase_alpha = 3
    nb_increase_alpha = 0
    for col in xrange(n):
        if adiag[col] == 0:
            alpha = max(alpha, alpha_min)

    success = False
    tired = False
    print 'colptrT:', colptrT

    colptr[:n + 1] = colptrT

    print 'colptr:', colptr

    while not (success or tired):

        # Store the scaled A into L.
        # Make room to store the computed factors.
        for k in xrange(nnzT):
            rowind[n_mem + k] = rowindT[k]

        for col in xrange(n + 1):
            colptr[col] = colptr[col] + n_mem
        for col in xrange(n):
            d[col] = adiag[col] * s[col] * s[col]
            if d[col] > 0:
                d[col] += alpha
            else:
                d[col] -= alpha

            for k in xrange(colptrT[col], colptrT[col + 1]):
                row = rowindT[k]
                lvals[n_mem + k] = valuesT[k] * s[col] * s[row]  # = S A S.

        # Attempt a factorization.
        success = attempt_lldl_INT64_FLOAT64(n, d, lvals, rowind, colptr, memory=memory)

        # Increase shift if the factorization didn't succeed.
        if not success:
            nb_increase_alpha += 1
            alpha *= 2

            tired = nb_increase_alpha > max_increase_alpha

            if not tired:
                colptr[:n + 1] = colptrT  # Reset colptr for the next round.

    nnzL = colptr[n]

    # Unscale L.
    for col in xrange(n):
        d[col] /= s[col] * s[col]
        for k in xrange(colptr[col], colptr[col + 1]):
            row = rowind[k]
            lvals[k] *= s[col] / s[row]

    return (np.asarray(colptr[:n + 1]),
            np.asarray(rowind[:nnzL]),
            np.asarray(lvals[:nnzL]),
            np.asarray(d), alpha)




@cython.profile(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef bint attempt_lldl_INT64_FLOAT64(unsigned int n,
                                      cnp.ndarray[cnp.float64_t, ndim=1] d,
                                      cnp.ndarray[cnp.float64_t, ndim=1] lvals,
                                      cnp.ndarray[cnp.int64_t, ndim=1] rowind,
                                      cnp.ndarray[cnp.int64_t, ndim=1] colptr,
                                      unsigned int memory=0):

    cdef int col, col_start, col_end, i, k, k1, kth, newk
    cdef int nzcol, row
    cdef unsigned int nz_to_keep
    cdef unsigned int kth_col_start, kth_col_end, new_col_start, new_col_end
    # cdef unsigned int n_mem = n * memory

    cdef cnp.float64_t lval

    # Work arrays.
    # contents of the current column of A.
    cdef cnp.ndarray[cnp.float64_t, ndim = 1] w = np.zeros(n, dtype=np.float64)
    # row indices of the nonzeros in the current column after it's been loaded
    # into w.
    cdef cnp.ndarray[cnp.int64_t, ndim = 1] indr = np.zeros(n, dtype=np.int64)
    # indf[col] = position in w of the next entry in column col to be used
    # during the factorization.
    cdef cnp.ndarray[cnp.int64_t, ndim = 1] indf = np.ones(n, dtype=np.int64)
    indf *= -1
    # llist[col] = linked list of columns that will update column col.
    cdef cnp.ndarray[cnp.int64_t, ndim = 1] llist = np.ones(n, dtype=np.int64)
    llist *= -1
    # cdef cnp.ndarray[cnp.int64_t, ndim=1] perm = np.zeros(n,
    # dtype=np.int64)

    # Attempt an incomplete LDL factorization.
    col_start = colptr[0]
    colptr[0] = 0

    # Scan each column in turn.
    for col in xrange(n):
        # Load column col of A into w.
        col_end = colptr[col + 1]

        nzcol = 0
        for k in xrange(col_start, col_end):
            row = rowind[k]
            w[row] = lvals[k]  # w[row] = A[row, col] for each col in turn.
            indr[nzcol] = row
            indf[row] = 0
            nzcol += 1

        # nzcol = number of nonzeros in current column.
        # (indr, w): sparse representation of current column:
        # - the nonzero element A[row, col] is in w[row], and
        # - the k-th nonzero element of A[:, col] is in w[indr[k]], (k = 0, ..., nzcol-1).

        # The factorization fails if the current pivot is zero.
        if d[col] == 0:
            print "Fail because of a null pivot"
            return False

        # Update column col using previous columns.
        k = llist[col]
        while k != -1:
            kth_col_start = indf[k]
            kth_col_end = colptr[k + 1] - 1

            lval = lvals[kth_col_start]  # lval = L[col, k].
            newk = llist[k]
            kth_col_start += 1

            if kth_col_start < kth_col_end:
                row = rowind[kth_col_start]
                indf[k] = kth_col_start
                llist[k] = llist[row]
                llist[row] = k

            # Perform the update L[row, col] <- L[row, col] - D[k, k] * L[col,
            # k] * L[row, k].
            for i in xrange(kth_col_start, kth_col_end + 1):
                row = rowind[i]
                if indf[row] != -1:
                    # w[row] = L[row, col], lval = L[col, k], lvals[i] = L[row,
                    # k].
                    w[row] -= d[k] * lval * lvals[i]
                else:
                    indf[row] = 0
                    indr[nzcol] = row
                    w[row] = -d[k] * lval * lvals[i]
                    nzcol += 1
            k = newk

        # Compute (incomplete) column col of L.
        for k in xrange(nzcol):
            row = indr[k]
            w[row] /= d[col]
            d[row] -= d[col] * w[row] * w[row]  # Variant I.

        nz_to_keep = min(col_end - col_start + memory, nzcol)
        kth = nzcol - nz_to_keep

        # FIX THIS
        # Determine the kth smallest elements in current column
        #       p = [1:nzcol]
        #       q = sortperm(w[indr[p]], by=abs)
        #       indr[p] = indr[p[q]]  # Permute this portion of indr.
        # kth_pos = indr[kth]   # Index of kth-smallest element in absolute
        # value in w.

        # Sort the row indices of the nz_to_keep largest elements.

        if kth >= 0:  # No need to sort if we keep all elements.
            perm = np.argpartition(np.fabs(w[indr[:nzcol]]), kth)
            # TODO: change sorting algorithm (default: quicksort)
            # temp = np.array(indr, dtype=np.int64)
            a = indr[perm[kth:]]
            a.sort()
            indr[:nzcol] = a[:]  # temp

        new_col_start = colptr[col]
        new_col_end = new_col_start + nz_to_keep
        for k in xrange(new_col_start, new_col_end):
            k1 = indr[kth + k - new_col_start]
            lvals[k] = w[k1]
            rowind[k] = k1

        # Variant II of diagonal elements update.
        #       for k = kth : nzcol
        #         d[indr[k]] -= diag[col] * w[indr[k]] * w[indr[k]]
        #       end

        if new_col_start < new_col_end - 1:
            indf[col] = new_col_start
            llist[col] = llist[rowind[new_col_start]]
            llist[rowind[new_col_start]] = col

        for k in xrange(nzcol):
            row = indr[k]
            indf[row] = -1
        col_start = colptr[col + 1]
        colptr[col + 1] = new_col_end

    colptr[n] = colptr[n - 1]
    return True


cdef cnp.int64_t flip_INT64(cnp.int64_t i):
    return -i - 2

cdef cnp.int64_t unflip_INT64(cnp.int64_t i):
    if i < 0:
        return flip_INT64(i)
    else:
        return i

cdef bint marked_INT64(cnp.int64_t * w, cnp.int64_t j):
    return w[j] < 0

cdef mark_INT64(cnp.int64_t * w, cnp.int64_t j):
    w[j] = flip_INT64(w[j])
    return

# cdef int reach(CSCSparseMatrix_INT64_t_FLOAT64_t G,
# CSCSparseMatrix_INT64_t_FLOAT64_t B, int k, INT64_t *xi, const int
# *pinv):
# , np.ndarray[cnp.int64_t, ndim=1] pinv):

cdef reach_INT64(cnp.int64_t n,
                    cnp.int64_t * Gi,
                    cnp.int64_t * Gr,
                    cnp.int64_t * Bi,
                    cnp.int64_t * Br,
                    int k,
                    cnp.ndarray[cnp.int64_t, ndim=1] xi):

    cdef:
        cnp.int64_t p
        cnp.int64_t top

    # TODO: check if xi is not null
    top = n
    print 'top:', top
    for p in xrange(Bi[k], Bi[k + 1]):
        # check inputs
        print "  p: %d, marked: %g" % (p, marked_INT64(Gi, Br[p]))
        if not marked_INT64(Gi, Br[p]):
            # <int *> cnp.PyArray_DATA(pinv)) # start a deep first sort at unmarked node i
            top = deep_first_sort_INT64_FLOAT64(Br[p], Gi, Gr, top,
                                                 < cnp.int64_t * > cnp.PyArray_DATA(xi),
                                                 < cnp.int64_t * > cnp.PyArray_DATA(xi) + n,
                                                 NULL)
            print 'dfs: ', top
            # sys.exit()

    for p in xrange(top, n):
        mark_INT64(Gi, xi[p])  # restore G
    return top


cdef cnp.int64_t deep_first_sort_INT64_FLOAT64(int j,
                                                          cnp.int64_t * Gi,
                                                          cnp.int64_t * Gr,
                                                          cnp.int64_t top,
                                                          cnp.int64_t * xi,
                                                          cnp.int64_t * pstack,
                                                          const cnp.int64_t * pinv):

    cdef:
        cnp.int64_t i
        cnp.int64_t p
        cnp.int64_t p2
        cnp.int64_t done
        cnp.int64_t jnew
        cnp.int64_t head = 0

    # TODO: check inputs
    # if (!xi || !pstack) return (-1) ;    /* check inputs */

    xi[0] = j  # initialize the recursion stack
    while head >= 0:
        j = xi[head]  # get j from the top of the recursion stack

        print '    j:', j
        if pinv != NULL:
            jnew = pinv[j]
        else:
            jnew = j

        print '    j,jnew: %d, marked: %g' % (jnew, marked_INT64(Gi, j))

        if not marked_INT64(Gi, j):
            print '    before mark: ', Gi[j]
            mark_INT64(Gi, j)  # mark node j as visited
            print '    after mark: ', Gi[j]
            if jnew < 0:
                pstack[head] = 0
            else:
                pstack[head] = unflip_INT64(Gi[jnew])

            print '    stack[head]:', pstack[head]

        done = 1  # node j done if no unvisited neighbors
        if jnew < 0:
            p2 = 0
        else:
            p2 = unflip_INT64(Gi[jnew + 1])

        for p in xrange(pstack[head], p2):  # examine all neighbors of j
            i = Gr[p]  # consider neighbor node i
            if marked_INT64(Gi, i):
                continue  # skip visited node i
            pstack[head] = p
            head += 1
            xi[head] = i
            done = 0
            break

        if done:
            head -= 1
            top -= 1
            xi[top] = j
    return top


# cpdef int sparse_solve_INT64_FLOAT64(cnp.int64_t n,
#                                       cnp.int64_t * Gi,
#                                       cnp.int64_t * Gr,
#                                       cnp.float64_t * Gv,
#                                       cnp.int64_t * Bi,
#                                       cnp.int64_t * Br,
#                                       cnp.float64_t * Bv,
#                                       int k,
#                                       cnp.ndarray[cnp.int64_t, ndim=1] xi,
#                                       cnp.ndarray[cnp.float64_t, ndim=1] x,
#                                       int lo):
#
#     cdef:
#         int j
#         int m
#         int J
#         int p
#         int q
#         int px
#         int top
#
#     cdef int *pinv = NULL
#
#     # TODO: check inputs
#     # if ( !xi || !x) return (-1) ;
#     Gi = G.ind ; Gr = G.row ; Gv = G.val ; n = G.nrow
#     Bi = B.ind ; Br = B.row ; Bv = B.val
#     print 'here'
#     top = reach_INT64(Gn, Gi, Gr, Bi, Br, k, <cnp.float64_t *> cnp.PyArray_DATA(xi) , pinv) # xi[top..n-1] = Reach(B(:,k))
#     print 'top:', top
#
#     # clear x
#     for p in xrange(top, n):
#         x[xi[p]] = 0
#
#     # scatter B
#     for p in xrange(Bi[k], Bi[k+1]):
#         x[Br[p]] = Bv[p]
#     for px in xrange(top, n):
#         j = xi[px]
#
#         if pinv != NULL:
#             J = pinv[j]
#         else:
#             J = j
#
#         if J < 0:
#             continue
#
#         if lo:
#             k = Gi[J]
#         else:
#             k = Gi[J+1] - 1
#
#         x[j] /= Gv[k] # x(j) /= G(j,j)
#
#         if lo:
#             p = Gi[J] +  1
#             q = Gi[J+1]
#         else:
#             p = Gi[J]
#             q = Gi[J+1]-1
#
#         for m in xrange(p, q):
#             x[Gr[m]] -= Gv[m] * x[j]
#
#     return top

cpdef dense_lsolve_INT64_FLOAT64(cnp.int64_t n,
                                  cnp.ndarray[cnp.int64_t, ndim=1] Li,
                                  cnp.ndarray[cnp.int64_t, ndim=1] Lr,
                                  cnp.ndarray[cnp.float64_t, ndim=1] Lv,
                                  cnp.ndarray[cnp.float64_t, ndim=1] x):

    cdef:
        cnp.int64_t p
        cnp.int64_t j

    # TODO: checks on inputs
    for j in xrange(n):
        x[j] /= Lv[Li[j]]
        for p in xrange(Li[j] + 1, Li[j + 1]):
            x[Lr[p]] -= Lv[p] * x[j]
    return

cpdef dense_ltsolve_INT64_FLOAT64(cnp.int64_t n,
                                   cnp.ndarray[cnp.int64_t, ndim=1] Li,
                                   cnp.ndarray[cnp.int64_t, ndim=1] Lr,
                                   cnp.ndarray[cnp.float64_t, ndim=1] Lv,
                                   cnp.ndarray[cnp.float64_t, ndim=1] x):

    cdef:
        cnp.int64_t p
        cnp.int64_t j

    # TODO: checks on inputs
    for j in xrange(n - 1, -1, -1):
        for p in xrange(Li[j] + 1, Li[j + 1]):
            x[j] -= Lv[p] * x[Lr[p]]
        x[j] /= Lv[Li[j]]
    return