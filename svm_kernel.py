import numpy as np
import numpy.linalg as la

class Kernel(object):

    @staticmethod
    def linear():
        # Implementing the linear method relation between two features x and y
        return lambda x,y:np.dot(x.T,y)

    @staticmethod
    def polykernel(dimension, offset):
        return lambda x, y: ((offset + np.dot(x.T,y)) ** dimension)

    @staticmethod
    def radial_basis(gamma):
        return lambda x, y: np.exp(-gamma*la.norm(np.subtract(x, y)))
