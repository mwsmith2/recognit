"""analyze.py: Contains functions for Principal Component Analysis (PCA)."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
import scipy.linalg as spl
import os
import random
import load
from collections import defaultdict


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

    return XTrain, XTest, trainlist, testlist


def calculateEig(X):
    """ Take  matrix and determine the first n principal components.

    Parameters
    ==========
    X : array_like

            The matrix which we will decompose into eigenvectors.

    Returns
    -------
    eigval : array_like

            List of eigenvalues in ascending order.

    eigvec : array_like

            List of eigenvectors in ascending order.

    """

    # Our data needs to be centered.
    for i in range(X.shape[1]):

        X[:, i] -= X[:, i].mean()

    # Get eigenvalues and vectors of cov(X_transpose) in ascending order.
    eigval, eigvec = spl.eigh(np.dot(X.T, X))

    # Convert these to eigenvalues of the cov(X).
    eigval = eigval / eigval.mean()
    eigvec = np.dot(eigvec, X.T)

    for i in range(eigvec.shape[0]):

        eigvec[i] = eigvec[i] / spl.norm(eigvec[i])

    # Return the largest n eigenvalues and vectors.
    return eigval[::-1], eigvec[::-1]


def characterizeFaces(XTrain, trainlist):

    nameDict = createDict(trainlist)
    eigval, eigvec = calculateEig(XTrain)

    eignorm = eigval.cumsum() / eigval.sum()

    neig = np.abs(eignorm - 0.90).argmin()
    weightDict = {}

    for name in nameDict:

        weightList = np.zeros(neig)

        for idx in nameDict[name]:

            weightList += np.dot(XTrain[:, idx], eigvec[:neig].T)

        weightDict[name] = weightList / len(nameDict[name])

    return weightDict, eigvec[:neig]


def calcWeight(image, eigvec):

    weight = np.dot(image, eigvec.T)

    return weight


def guessWho(weightDict, weight):

    proximity = {}

    for name in weightDict:

        proximity[name] = spl.norm(weightDict[name] - weight)

    person = min(proximity, key=proximity.get)

    return person


def createDict(filelist):

    nameDict = defaultdict(list)

    for i in range(len(filelist)):

        nameDict[filelist[i].split('_')[0]].append(i)

    return nameDict
