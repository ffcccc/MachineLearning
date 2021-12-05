# distutils: language = c++
# distutils: sources = cpp/distance.cpp

from eigency.core cimport *
# cimport eigency.conversions
# from cpp.eigency cimport *


# import eigency
# include "../eigency.pyx"

cdef extern from "distance.h":
     cdef double _distEuclidean "distEuclidean"(Map[ArrayXd] &, Map[ArrayXd] &)

# This will be exposed to Python
def distEuclidean(np.ndarray arrayX, np.ndarray arrayY):
    return _distEuclidean(Map[ArrayXd](arrayX), Map[ArrayXd](arrayY))