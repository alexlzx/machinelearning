import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import utils
from kmeans import Kmeans
from kmedians import Kmedians
from quantize_image import ImageQuantizer
from sklearn.cluster import DBSCAN
import linear_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1', '1.1', '1.2', '1.3', '1.4', '2', '2.2', '4', '4.1', '4.3'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1':
        X = utils.load_dataset('clusterData')['X']

        model = Kmeans(k=4)
        model.fit(X)
        utils.plot_2dclustering(X, model.predict(X))
        
        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    if question == '1.1':
        X = utils.load_dataset('clusterData')['X']
        min_error = np.inf
        min_pred_y = np.zeros(X.shape[0])

        for i in range(50):
            model = Kmeans(k=4)
            model.fit(X)
            y_pred = model.predict(X)
            error = model.error(X)

            if error < min_error:
                min_error = error
                min_pred_y = y_pred

        utils.plot_2dclustering(X, min_pred_y)
        print("The error is: %.3f" % min_error)

        fname = os.path.join("..", "figs", "1.1_kmeans_lowest_err.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    if question == '1.2':
        X = utils.load_dataset('clusterData')['X']
        min_errors = np.zeros(10)
        kk = np.arange(1,11)

        for k in kk:
            min_error = np.inf
            min_pred_y = np.zeros(X.shape[0])

            for i in range(50):
                model = Kmeans(k)
                model.fit(X)
                y_pred = model.predict(X)
                error = model.error(X)

                if error < min_error:
                    min_error = error

            min_errors[k-1] = min_error

        plt.plot(kk, min_errors)
        plt.xlabel("Number of clusters")
        plt.ylabel("Error")
        plt.title("Min error found across 50 random initializations")

        fname = os.path.join("..", "figs", "1.2_kmeans_err_with_k.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    if question == '1.3':
        X = utils.load_dataset('clusterData2')['X']

        # Q 1.3.1
        min_error = np.inf
        min_pred_y = np.zeros(X.shape[0])

        for i in range(50):
            model = Kmeans(k=4)
            model.fit(X)
            y_pred = model.predict(X)
            error = model.error(X)

            if error < min_error:
                min_error = error
                min_pred_y = y_pred

        utils.plot_2dclustering(X, min_pred_y)
        print("The error is: %.3f" % min_error)

        fname = os.path.join("..", "figs", "1.3.1_kmeans_outliers.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        # Q 1.3.2
        min_errors = np.zeros(10)
        kk = np.arange(1,11)

        for k in kk:
            min_error = np.inf
            min_pred_y = np.zeros(X.shape[0])

            for i in range(50):
                model = Kmeans(k)
                model.fit(X)
                y_pred = model.predict(X)
                error = model.error(X)

                if error < min_error:
                    min_error = error

            min_errors[k-1] = min_error

        plt.figure()
        plt.plot(kk, min_errors)
        plt.xlabel("Number of clusters")
        plt.ylabel("Error")
        plt.title("Min error found across 50 random initializations")

        fname = os.path.join("..", "figs", "1.3.2_kmeans_err_with_outlier.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        # Q 1.3.4
        min_error = np.inf
        min_pred_y = np.zeros(X.shape[0])

        for i in range(50):
            model = Kmedians(k=4)
            model.fit(X)
            y_pred = model.predict(X)
            error = model.error(X)

            if error < min_error:
                min_error = error
                min_pred_y = y_pred

        plt.figure()
        utils.plot_2dclustering(X, min_pred_y)
        print("The error is: %.3f" % min_error)

        fname = os.path.join("..", "figs", "1.3.4_kmedians_outliers.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    if question == '1.4':
        X = utils.load_dataset('clusterData2')['X']
        
        model = DBSCAN(eps=2, min_samples=3)
        y = model.fit_predict(X)

        utils.plot_2dclustering(X,y)

        fname = os.path.join("..", "figs", "1.4.3_clusterdata_4_dbscan.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    if question == '2':
        img = utils.load_dataset('dog')['I']/255
        bb = np.array([1,2,4,6])

        for b in bb:
            model = ImageQuantizer(b)
            quantize_image, mean = model.quantize(img)
            compressed_image = model.dequantize(quantize_image, mean)

            plt.imshow(compressed_image)
            fname = os.path.join("..","figs","2_quantized_dog_{}.pdf".format(b))
            plt.savefig(fname)
            print("\nFigure saved as '%s'" % fname)


    elif question == "4":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        # Draw model prediction
        Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample,yhat,'g-', label = "Least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join("..","figs","least_squares_outliers.pdf")
        print("Saving", figname)
        plt.savefig(figname)

    elif question == "4.1":
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Fit least-squares estimator
        model = linear_model.WeightedLeastSquares()
        # setting z = 1 for the first 400 data points and z = 0.1 for the last 100 data points
        z = np.concatenate(([1]*400, [0.1]*100), axis = 0)
        model.fit(X,y,z)
        print(model.w)

        # Draw model prediction
        Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample,yhat,'g-', label = "Weighted Least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join("..","figs","weighted_least_squares_outliers.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "4.3":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Draw model prediction
        Xsample = np.linspace(np.min(X), np.max(X), 1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample, yhat, 'g-', label = "Least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join("..","figs","gradient_descent_model.pdf")
        print("Saving", figname)
        plt.savefig(figname)
