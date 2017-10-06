import sys
import argparse
import os
import numpy as np
import utils
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor as NeuralNet
from sklearn.datasets import load_boston

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True, choices=['1', '2'])

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == '1':

        data = utils.load_dataset('basisData')
        
        X = data['X']
        y = data['y'].ravel()
        Xtest = data['Xtest']
        ytest = data['ytest'].ravel()
        n,d = X.shape
        t = Xtest.shape[0]
        
        model = NeuralNet(
                        solver="lbfgs",
                        hidden_layer_sizes=(10),
                        alpha=0.5)
        model.fit(X,y)
        
        # Comput training error
        yhat = model.predict(X)
        trainError = np.mean((yhat - y)**2)
        print("Training error = ", trainError)
        
        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest)**2)
        print("Test error     = ", testError)
        
        plt.figure()
        plt.plot(X, y, 'b.', label="training data", markersize=2)
        plt.title('Training Data')
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot()
        plt.plot(Xhat, yhat, 'g', label="neural network")
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join("..","figs","basisData.pdf")
        print("Saving", figname)
        plt.savefig(figname)
        
    elif question == '2':
        X = load_boston().data
        y = load_boston().target

        n, d = X.shape

        # Split training data into a training and a validation set
        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]

        model = NeuralNet(
                        solver="lbfgs",
                        hidden_layer_sizes=(10),
                        alpha=0.1)

        model.fit(X,y)

        # Compute validation error
        yhat = model.predict(Xvalid)
        validError = np.sum((yhat - yvalid)**2)/ (n//2)
        print("Validation error = {}".format(validError))
