import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

import utils
from pca import PCA, AlternativePCA, RobustPCA
from manifold import MDS, ISOMAP
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from naive_bayes import NaiveBayes

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == '2.2':
        # 1. Load dataset
        dataset = utils.load_dataset("newsgroups")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        
        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])

        # 2. Evaluate the decision tree model with depth 20
        model = RandomForestClassifier()
        model.fit(X, y)
        y_pred = model.predict(X_valid)

        v_error = np.mean(y_pred != y_valid)
        print("Random Forest Validation error: %.3f" % v_error)

        # 3. Evaluate the Naive Bayes Model
        model = NaiveBayes(num_classes=4)
        print("Fitting...")
        model.fit(X, y)

        print("Predicting...")
        y_pred = model.predict(X_valid)

        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes Validation error: %.3f" % v_error)

        # This should print a validation error of 0.19

        
    if question == '3.2':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        # standardize columns
        X = utils.standardize_cols(X)

        # Plot the matrix
        plt.imshow(X)
        utils.savefig('unsatisfying_visualization_1.png')

        ## Randomly plot two features, and label all points
        ## my code starting here ................
        model = PCA(k=2)
        model.fit(X)
        Z = model.compress(X)
        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('PCA')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0],Z[i,1]))
        utils.savefig('Q3.2 PCA.png')

        # Q3.3
        # variance is explained by 
        # Frobenius norm
        variance_remains = norm(model.expand(Z) - X)**2 / norm(X)**2
        variance_explained = 1 - variance_remains
        print("Variance is explained by: "+ "{:0.4f}".format(variance_explained))


    if question == '4':
        X = utils.load_dataset('highway')['X'].astype(float)/255
        n,d = X.shape
        h,w = 64,64 # height and width of each image

        # the two variables below are parameters for the foreground/background extraction method
        # you should just leave these two as default.

        k = 5 # number of PCs
        threshold = 0.04 # a threshold for separating foreground from background

        # model = AlternativePCA(k=k)
        # model.fit(X)
        # Z = model.compress(X)
        # Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        for i in range(10):
            plt.subplot(2,3,1)
            plt.title('Original')
            plt.imshow(X[i].reshape(h,w).T, cmap='gray')
            plt.subplot(2,3,2)
            plt.title('PCA Reconst.') # Reconstruction
            plt.imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')
            plt.subplot(2,3,3)
            plt.title('PCA Thresh. Diff.') # Thresholded Difference
            plt.imshow(1.0*(abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray')

            plt.subplot(2,3,4)
            plt.title('Original')
            # plt.imshow(X[i].reshape(h,w).T, cmap='gray')
            plt.subplot(2,3,5)
            plt.title('RPCA Reconst.')
            # plt.imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')
            plt.subplot(2,3,6)
            plt.title('RPCA Thresh. Diff.')
            # plt.imshow(1.0*(abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray')

            utils.savefig('Q4. highway_{:03d}.jpg'.format(i))


    if question == '5':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        model = MDS(n_components=2)
        Z = model.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0], Z[i,1]))
        utils.savefig('MDS_animals.png')

    if question == '5.1':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        for n_neighbours in [2,3]:
            model = ISOMAP(n_components=2, n_neighbours=n_neighbours)
            Z = model.compress(X)

            fig, ax = plt.subplots()
            ax.scatter(Z[:,0], Z[:,1])
            plt.ylabel('z2')
            plt.xlabel('z1')
            plt.title('ISOMAP with NN=%d' % n_neighbours)
            for i in range(n):
                ax.annotate(animals[i], (Z[i,0], Z[i,1]))
            utils.savefig('ISOMAP%d_animals.png' % n_neighbours)
