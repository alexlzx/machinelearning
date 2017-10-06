import sys
import argparse
import linear_model
import matplotlib.pyplot as plt
import numpy as np
import utils
import os
from sklearn.utils import shuffle

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        # Load the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2) / n
        print("Training error = ", trainError)

        # Compute test error

        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2) / t
        print ("Test error = ", testError)

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.', label = "Training data")
        plt.title('Training Data')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g', label = "Least squares fit")
        plt.legend(loc="best")
        figname = os.path.join("..","figs","leastSquares.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "1.1":
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        ''' YOUR CODE HERE'''
        # Fit least-squares model
        model = linear_model.LeastSquaresBias()
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2) / n
        print("Training error = ", trainError)

        # Compute test error

        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2) / t
        print ("Test error = ", testError)

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.', label = "Training data")
        plt.title('Training Data')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g', label = "Least squares bias fit")
        plt.legend(loc="best")
        figname = os.path.join("..","figs","Q1.1 leastSquaresBias.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "1.2":

        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        for p in range(11):
            print("p=%d" % p)

            ''' YOUR CODE HERE '''
            # Fit least-squares model
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)

            # Compute training error
            yhat = model.predict(X)
            trainError = np.sum((yhat - y)**2) / n
            print("Training error = ", trainError)

            # Compute test error
            yhat = model.predict(Xtest)
            testError = np.sum((yhat - ytest)**2) / t
            print ("Test error = ", testError)

            # Plot model
            plt.figure()
            plt.plot(X,y,'b.', label = "Training data")
            plt.title('Training Data. p = {}'.format(p))
            # Choose points to evaluate the function
            Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]


            '''YOUR CODE HERE'''
            #Predict on Xhat
            yhat = model.predict(Xhat)
            plt.plot(Xhat,yhat,'g', label = "Least squares poly fit")

            plt.legend()
            figname = os.path.join("..","figs","PolyBasis%d.pdf"%p)
            print("Saving", figname)
            plt.savefig(figname)


    elif question == "2.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
        t = Xtest.shape[0]

        # Split training data into a training and a validation set
        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set

        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            # Train on the training set
            model = linear_model.LeastSquaresRBF(sigma)
            model.fit(Xtrain,ytrain)

            # Compute the error on the validation set
            yhat = model.predict(Xvalid)
            validError = np.sum((yhat - yvalid)**2)/ (n//2)
            print("Error with sigma = {:e} = {}".format( sigma ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title('Training Data')

        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g',label = "Least Squares with RBF kernel and $\sigma={}$".format(bestSigma))
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join("..","figs","least_squares_rbf.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "2.2":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
        t = Xtest.shape[0]

        ''' YOUR CODE HERE '''
        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s
            total_err = 0
            X_shuffled, y_shuffled = shuffle(X, y, random_state=0)
            fold_size = X_shuffled.shape[0]//10

            for i in range(10):
                v_start_index = fold_size*i
                v_end_index = fold_size*(i+1)
                X_validation = X_shuffled[v_start_index:v_end_index]
                y_validation = y_shuffled[v_start_index:v_end_index]
                
                t_start_index = 0
                t_end_index = X_shuffled.shape[0]
                X_train = X_shuffled[np.r_[t_start_index:v_start_index, v_end_index:t_end_index]]
                y_train = y_shuffled[np.r_[t_start_index:v_start_index, v_end_index:t_end_index]]

                model = linear_model.LeastSquaresRBF(sigma)
                model.fit(X_train,y_train)

                yhat = model.predict(X_validation)
                validError = np.sum((yhat - y_validation)**2)/ (n//10)
                total_err += validError
                print("Error with sigma = {:e} = {}".format( sigma ,validError))

            validError = total_err / 10
            
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title('Training Data')

        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g',label = "Least Squares with RBF kernel and $\sigma={}$".format(bestSigma))
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join("..","figs","Q2.2-least_squares_rbf_CV.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "3":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logReg(maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogReg Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logReg Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())


    elif question == "3.1":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL2 model
        model = linear_model.logRegL2(lammy=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "3.2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL1 model
        model = linear_model.logRegL1(L1_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())


    elif question == "3.3":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL0 model
        model = linear_model.logRegL0(L0_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nTraining error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())


    elif question == "4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Fit One-vs-all Least Squares
        model = linear_model.leastSquaresClassifier()
        model.fit(XMulti, yMulti)

        print("leastSquaresClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("leastSquaresClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

        print(np.unique(model.predict(XMultiValid)))

    elif question == "4.1":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']
        
        # Fit One-vs-all Logistic Regression
        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)

        print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))


    elif question == "4.4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Fit logRegL2 model
        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

    else:
        print("Not a valid question number.")
