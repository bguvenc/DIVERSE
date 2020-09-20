"""
Methods for generating mask matrices, with 1 entries indicating observed and 0
indicating unobserved.
Provide methods for single mask matrices, and cross-validation folds.
"""


import numpy
import itertools


def nonzero_indices(M):
    (I,J) = numpy.array(M).shape
    return [(i,j) for i,j in itertools.product(range(I),range(J)) if M[i][j] != 0]

def zero_indices(M):
    (I,J) = numpy.array(M).shape
    return [(i,j) for i,j in itertools.product(range(I),range(J)) if M[i][j] == 0]


