import numpy as np
from numpy.linalg import solve
import findMin
import sys
from scipy.optimize import approx_fprime


# Original Least Squares
class LeastSquares:
    # Class constructor
    def __init__(self):
        pass

    def fit(self,X,y):
        # Solve least squares problem

        a = np.dot(X.T, X)
        b = np.dot(X.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):

        w = self.w
        yhat = np.dot(Xhat, w)
        return yhat

# Least Squares where each sample point X has a weight associated with it.
class WeightedLeastSquares:

    def __init__(self):
        pass

    def fit(self,X,y,z):

        ''' YOUR CODE HERE '''
        Z = np.diag(z)

        # w = np.linalg.solve(a, b) where aw = b
        # a = X.T * Z * X
        a = np.dot(np.dot(X.T, Z), X)

        # b = X.T * Z * y
        b = np.dot(np.dot(X.T, Z), y)

        self.w = solve(a, b)


    def predict(self,Xhat):
        '''YOUR CODE HERE '''
        w = self.w
        yhat = np.dot(Xhat, w)
        return yhat

class LinearModelGradient:

    def __init__(self):
        pass

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin.findMin(self.funObj, self.w, 100, X, y)

    def predict(self,Xtest):

        w = self.w
        yhat = Xtest*w
        return yhat

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''

        # Calculate the function value
        # f = sum of log(exp(w^T xi - yi) + exp(yi - w^T xi))
        f = np.sum(np.log(np.exp(X.dot(w) - y) + np.exp(y - X.dot(w))))

        # Calculate the gradient value
        # graident of log-sum-exp approximation
        g = X.T.dot((np.exp(X.dot(w) - y) - np.exp(y - X.dot(w))) / (np.exp(X.dot(w) - y) + np.exp(y - X.dot(w))))

        return f,g



