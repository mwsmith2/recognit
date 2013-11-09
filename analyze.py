"""analyze.py: Contains functions for Principal Component Analysis (PCA)."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
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

    with open(filename) as f:

        filelist = f.readlines()

    nimages = len(filelist)

    random.seed(sd)
    trainlist = random.sample(filelist, int(nimages * ratio))
    testlist = [x for x in filelist if x not in trainlist]

    for i in range(len(trainlist)):

        if i == 0:

            image = load.readPGM(trainlist(i))

            width = image.shape[0]
            height = image.shape[1]

            XTrain = np.empty((width * height, len(trainlist)))

            XTrain[:, i] = image.reshape(-1)

        else:

            image = load.readPGM(trainlist(i))
            XTrain[:, i] = image.reshape(-1)

    for j in range(len(testlist)):

        if j == 0:

            image = load.readPGM(testlist(j))

            width = image.shape[0]
            height = image.shape[1]

            XTest = np.empty((width * height, len(testlist)))

            XTest[:, j] = image.reshape(-1)

        else:

            image = load.readPGM(testlist(j))
            XTest[:, j] = image.reshape(-1)

    return XTrain, XTest