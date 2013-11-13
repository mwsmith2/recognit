"""analyze.py: Contains functions for Principal Component Analysis (PCA)."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
import scipy.linalg as spl
from collections import defaultdict
from scipy.spatial.distance import euclidean
from sklearn import neighbors


def calculateEig(X):
    """ Take matrix and determine eigenvectors and eigenvalues.

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

    # Calculate the mean face.
    meanface = X.mean(axis=-1)

    for i in range(X.shape[1]):

        X[:, i] = X[:, i] - meanface

    # Get eigenvalues and vectors of cov(X.T * X) in ascending order.
    eigval, eigvec = spl.eigh(np.dot(X.T, X))

    # Convert these to eigenvalues of the cov(X).
    eigvec = np.dot(X, eigvec)

    for i in range(eigvec.shape[1]):

        eigvec[:, i] = eigvec[:, i] / spl.norm(eigvec[:, i])

    # Return the largest n eigenvalues and vectors.
    return eigval[::-1], eigvec[:, ::-1]


def characterizeFaces(XTrain, trainlist, threshold=0.95):
    """ Take matrix and determine the first n principal components.
        Calculate weights for different features and match them with strings.

    Parameters
    ==========
    XTrain : array_like

            The matrix which we will decompose into eigenvectors.

    trainlist : string_list

            List of strings that contains names of features.

    Returns
    -------
    weightDict : dict

            Weights for each feature matched with the keyword.

    eigvec : array_like

            List of principal eigenvectors in ascending order.

    """

    featureDict = createDict(trainlist)  # Create a feature dictionary.
    eigval, eigvec = calculateEig(XTrain)  # Calculate eigenvectors.

    meanface = XTrain.mean(axis=-1)  # Calculate mean face.

    eignorm = eigval.cumsum() / eigval.sum()  # Normalize them.

    neig = np.abs(eignorm - threshold).argmin()  # Select the principal ones.
    weightDict = {}

    # Calculate weights for every feature in the dictionary.
    for feature in featureDict:

        weights = {}
        i = 0

        for idx in featureDict[feature]:

            XTrain[:, idx] = XTrain[:, idx] - meanface
            weights[i] = np.dot(eigvec[:, :neig].T, XTrain[:, idx])
            i += 1

        weightDict[feature] = weights

    return weightDict, eigvec[:, :neig], meanface


def guessDistance(weightDict, weight):
    """ Gets weight dictionary and weights for a single image and
        guesses the label for asked feature.

    Parameters
    ==========
    weightDict : dict

            Weights for each feature matched with the keyword.

    weight : array_like

            Weights for given image.

    Returns
    -------
    label : string

            Guessed label for the feature.

    distance : float

            Distance to the closest feature.

    """
    distance = {}

    for feature in weightDict:

        d = np.zeros(len(weight))

        for idx in weightDict[feature]:

            d += euclidean(weightDict[feature][idx], weight)

        distance[feature] = d / len(weightDict[feature])

    label = min(distance, key=distance.get)

    return label


def guesskNN(weightDict, weight):

    labelDict = {}
    i = 0

    size = sum(len(feature) for feature in weightDict.itervalues())

    for feature in weightDict:

        labelDict[feature] = i
        i += 1

    x = np.empty((size, len(weight)))
    y = np.empty(size)
    i = 0

    for feature in weightDict:

        for idx in weightDict[feature]:

            x[i, :] = weightDict[feature][idx]
            y[i] = labelDict[feature]
            i += 1

    clf = neighbors.KNeighborsClassifier(3)
    clf.fit(x, y)
    p = clf.predict(weight)

    guess = [label for label, value in labelDict.items() if value==p[0]][0]

    return guess


def calcWeight(image, eigvec, meanface):
    """Calculate weights for a given image."""

    image = image - meanface

    weight = np.dot(eigvec.T, image)

    return weight


def createDict(filelist, n=0):
    """Create a dictionary for feature given by n, using filenames."""

    nameDict = defaultdict(list)

    for i in range(len(filelist)):

        nameDict[filelist[i].split('_')[n]].append(i)

    return nameDict
