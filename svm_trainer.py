import cvxopt.solvers
import numpy as np

from svm_predictor import SVMPredictor

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class SVMTrainer(object):

    def __init__(self, kernel, c):
        # Assigning the attributes kernal and C value
        self.kernel = kernel
        self.c = c

    def train(self, X, y):
        # Training function
        # Caluculate the langrange multipliers
        lagrange_multipliers = self.compute_multipliers(X, y)
        # Trainer returns SVM predictor that is used to predict which class the test element belongs to
        return self.construct_predictor(X, y, lagrange_multipliers)

    def kernel_matrix(self, X, n_samples):
        # Size of kernal matrix is (no_of_inputs , no_of_inputs)
        # Reason for this is that kernel function value is calculated between every 2 inputs given
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
        #  Returns the kernel function values
        return K

    def construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]
        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self.kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])
        return SVMPredictor(
            kernel=self.kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def compute_multipliers(self, X, y):
        # n_samples is no_of_inputs
        # n_features is no_of_features
        n_samples, n_features = X.shape
        # Returns kernel function matrix
        K = self.kernel_matrix(X,n_samples)
        # np.outer(a,b) gives
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        # Create a diagonal matrix of (n_samples , n_samples) dimension with -1 as value
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.c)
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        A = cvxopt.matrix(y, (1, n_samples) , 'd')
        b = cvxopt.matrix(0.0)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # To flatten as one dimension array
        return np.ravel(solution['x'])
