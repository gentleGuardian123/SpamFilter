import numpy as np

class SVMPredictor(object):

    # Initializing the SVM predictor
    # Needs kernel (K) , bias (b) , weights (w) , support_vectors and support_vector_labels
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        # Weights is equal to support vector labels as per mathematical formulation
        assert len(weights) == len(support_vector_labels)

    def predict(self, x):
        # result = b
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            # result += w * support_vector_labels * K
            result += z_i * y_i * self._kernel(x_i, x)
        # Returning the sign of the value predicted
        # +1 means belonging to positive (non spam) class
        # -1 means belonging to negative (spam) class
        return np.sign(result).item()