import numpy as np
import utils

class Kmedians:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        mid = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            mid[kk] = X[i]

        while True:
            y_old = y

            # Compute L1 Norm distance to each mean
            dist2 = np.zeros((N, self.k))

            for i in range(N):
                for k_val in range(self.k):
                    dist2[i][k_val] = np.sum(np.abs(X[i] - mid[k_val])) 

            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update mid
            for kk in range(self.k):
                mid[kk] = np.median(X[y==kk], axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-medians, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.mid = mid

    def predict(self, X):
        mid = self.mid
        N = X.shape[0]
        dist2 = np.zeros((N, self.k))

        for i in range(N):
            for k_val in range(self.k):
                dist2[i][k_val] = np.sum(np.abs(X[i] - mid[k_val])) 

        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        mid = self.mid
        y_pred = self.predict(X)
        err = 0

        for k_val in range(self.k):
            X_kval = X[y_pred==k_val]
            for i in range(X_kval.shape[0]):
                err += np.sum(np.abs(X_kval[i] - mid[k_val])) 

        return err
