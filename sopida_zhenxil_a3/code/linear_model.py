import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

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

# Least Squares with a bias added
class LeastSquaresBias:
    def __init__(self):
        pass

    def fit(self,X,y):
        n = X.shape[0]       # rows
        v = np.ones((n, 1))  # column of ones
        Z = np.c_[v, X]      # new matrix Z with a column of ones on the left

        a = np.dot(Z.T, Z)
        b = np.dot(Z.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):
        w = self.w
        n = Xhat.shape[0]      
        v = np.ones((n, 1))
        Zhat = np.c_[v, Xhat]
        yhat = np.dot(Zhat, w)
        return yhat

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        a = np.dot(Z.T, Z)
        b = np.dot(Z.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):
        w = self.w
        Zhat = self.__polyBasis(Xhat)
        yhat = np.dot(Zhat, w)
        return yhat

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):

        n = X.shape[0]
        d = self.p + 1
        # Z should have as many rows as X and as many columns as (p+1)
        Z = np.ones((n, d))

        ''' YOUR CODE HERE FOR Q1.2'''
        for col in range(1, d):
            for row in range(0, n):
                Z[row][col] = X[row][0] ** col

        return Z

# Least Squares with RBF Kernel
class LeastSquaresRBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self,X,y):
        self.X = X
        [n, d] = X.shape

        Z = self.__rbfBasis(X, X, self.sigma)

        # Solve least squares problem
        l = 1e-12

        a = Z.T.dot(Z) + l* np.identity(n)
        b = np.dot(Z.T, y)
        self.w = solve(a,b)

    def predict(self, Xtest):
        Z = self.__rbfBasis(Xtest, self.X, self.sigma)
        yhat = Z.dot(self.w)
        return yhat

    def __rbfBasis(self, X1, X2, sigma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        den = 1 / np.sqrt(2 * np.pi * (sigma** 2))

        D = (X1**2).dot(np.ones((d, n2))) + \
            (np.ones((n1, d)).dot((X2.T)** 2)) - \
            2 * (X1.dot( X2.T))

        Z = den * np.exp(-1* D / (2 * (sigma**2)))
        return Z


class logReg:
    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

class logRegL2:
    # L2 Regularized Logistic Regression
    def __init__(self, lammy=1.0, verbose=1, maxEvals=100):   # lammy is the L2 lambda 
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))
        f += (self.lammy / 2) * np.sum(w ** 2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w 

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

class logRegL1:
    # L1 Regularized Logistic Regression
    def __init__(self, L1_lambda=1.0, verbose=1, maxEvals=100):   # lammy is the L2 lambda 
        self.verbose = verbose
        self.L1_lambda = L1_lambda
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # add "i" to the set
                # TODO for Q3.3: Fit the model with 'i' added to the features,
                # then compute the score and update the minScore/minInd
                w, loss = minimize(list(selected_new))
                loss += self.L0_lambda * len(selected_new)
                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    # a silly classifier that uses least squares
    def __init__(self):
        pass

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            self.W[:, i] = np.linalg.lstsq(np.dot(X.T, X), np.dot(X.T, ytmp))[0]

    def predict(self, X):
        yhat = np.dot(X, self.W)

        return np.argmax(yhat, axis=1)


class logLinearClassifier:
    def __init__(self, maxEvals, verbose):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        f = np.sum(np.log(1. + np.exp(-yXw)))

        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            (self.W[:, i],f) = findMin.findMin(self.funObj, self.W[:, i],
                                            self.maxEvals, X, ytmp, verbose=self.verbose)

    def predict(self, X):
        yhat = np.dot(X, self.W)

        return np.argmax(yhat, axis=1)


class softmaxClassifier:
    def __init__(self, maxEvals):
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        n, d = X.shape

        y_bool = np.zeros((n,self.n_classes)).astype(bool)
        y_bool[range(n), y] = 1

        W = np.reshape(w, (d, self.n_classes))
        XW = np.dot(X, W)
        E = np.sum(np.exp(XW), axis=1)

        f = np.sum(-XW[y_bool] + np.log(E))
        g = X.T.dot(np.exp(XW) / np.reshape(E, (E.shape[0], 1)) - y_bool)

        return f, g.flatten()

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros(d*self.n_classes)
        utils.check_gradient(self, X, y)

        (self.w, f) = findMin.findMin(self.funObj, self.w.flatten(),
                                      self.maxEvals, X, y)

        self.w = np.reshape(self.w, (d, self.n_classes))

    def predict(self, X):
        yhat = np.dot(X, self.w)

        return np.argmax(yhat, axis=1)