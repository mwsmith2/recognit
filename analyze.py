"""analyze.py: Contains functions for Principal Component Analysis (PCA)."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
import scipy.linalg as la
import os
import random
import load


def createMatrices(filename, sd=1234, ratio=0.6):
    """
    Creates matrices for training and test.

    Parameters
    ----------
    filename : path

            Path to .txt file that includes list of PGM images.

    sd : int

            Seed for random set generation. The default is 1234.

    ratio : float

            Ratio of images in training set to total number. Default is 0.6.

    Returns
    -------
    XTrain : array_like

            Matrix of PGM images. Every column corresponds to an image.

    XTest : array_like

            Matrix of PGM images. Every column corresponds to an image.

    """

    datadir = os.path.dirname(os.path.abspath(filename))

    with open(filename) as f:

        filelist = f.readlines()


    nimages = len(filelist)

    random.seed(sd)
    trainlist = random.sample(filelist, int(nimages * ratio))
    testlist = [x for x in filelist if x not in trainlist]

    for i in range(len(trainlist)):

        if i == 0:

            image = load.readPGM(datadir + '/' + trainlist[i].rstrip('\n'))

            width = image.shape[0]
            height = image.shape[1]

            XTrain = np.empty((width * height, len(trainlist)))

            XTrain[:, i] = image.reshape(-1)

        else:

            image = load.readPGM(datadir + '/' + trainlist[i].rstrip('\n'))
            XTrain[:, i] = image.reshape(-1)

    for j in range(len(testlist)):

        if j == 0:

            image = load.readPGM(datadir + '/' + testlist[j].rstrip('\n'))

            width = image.shape[0]
            height = image.shape[1]

            XTest = np.empty((width * height, len(testlist)))

            XTest[:, j] = image.reshape(-1)

        else:

            image = load.readPGM(datadir + '/' + testlist[j].rstrip('\n'))
            XTest[:, j] = image.reshape(-1)

    return XTrain, XTest


def PCA(X, n=10):
    """ Take a matrices and determine the first n principal components.

    Parameters
    ==========
    X : array_like

            The matrix which we will decompose into eigenvectors.

    n : int

            The number of eigenvectors to be returned.

    """

    # Our data needs to be centered
    for i in range(X.shape[1]):

        X[:, i] -= np.mean(X[:, i])

    # Get eigenvalues and vectors of cov(X_transpose) in ascending order
    lamb0, evec0 = la.eigh(np.dot(np.transpose(X), X))

    # Convert these to eigenvalues of the cov(X)
    lamb = lamb0 / np.mean(lamb0)
    evec = np.dot(evec0, np.transpose(X))
    for i in range(evec.shape[0]):
        evec[i] = evec[i] / np.mean(evec[i])

    print lamb.shape
    print lamb
    print evec.shape
    print evec

    # Return the largest n eigenvalues/vectors
    return lamb[-n:], evec[-n:]