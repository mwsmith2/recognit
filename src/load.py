"""load.py: Contains functions to load PGM file types and create datasets."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
import os
import random
from matplotlib.image import imread


def readPGM(filename):
    """
    PGM to numpy.ndarray converter.

    Parameters
    ----------
    filename : path

            Path to .pgm file.

    Returns
    -------
    image : array_like

            PGM file returned as numpy.ndarray with type int8 or int16.

    """

    f = open(filename)
    pgmID = f.readline()  # This is PGM identifier.

    if pgmID[:2] == 'P2':

        image = decodeP2(f)
        f.close()

        return image

    elif pgmID[:2] == 'P5':

        f.close()
        image = decodeP5(filename)


        return image

    else:

        print "Not a valid PGM file!"

    return -1


def decodeP2(f):
    """
    Decode P2 type PGM files.

    Parameters
    ----------
    f : file handler

            Handler of .pgm file.

    Returns
    -------
    image : array_like

            PGM file returned as numpy.ndarray with type int8.

    """
    line = f.readline().split()  # This line contains width and height.
    width = int(line[0])
    height = int(line[1])

    line = f.readline().split()  # The maxval which we don't need.

    image = np.zeros([height, width], dtype='int8')

    for n in range(height):

        values = []

        while (len(values) < width and len(line) != 0):

            line = f.readline().split()

            for x in line:

                values.append(np.int(x))

        image[n] = np.array(values)

    return image


def decodeP5(filename):
    """
    Decode P5 type PGM files.

    Parameters
    ----------
    filename : filename

            Filename of P5 type file.

    Returns
    -------
    image : array_like

            PGM file returned as numpy.ndarray.

    """

    image = imread(filename)

    return image


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

            image = readPGM(datadir + '/' + trainlist[i].rstrip('\n'))

            height = image.shape[0]
            width = image.shape[1]

            XTrain = np.empty((width * height, len(trainlist)))

            XTrain[:, i] = image.reshape(-1)

        else:

            image = readPGM(datadir + '/' + trainlist[i].rstrip('\n'))
            XTrain[:, i] = image.reshape(-1)

    for j in range(len(testlist)):

        if j == 0:

            image = readPGM(datadir + '/' + testlist[j].rstrip('\n'))

            height = image.shape[0]
            width = image.shape[1]

            XTest = np.empty((width * height, len(testlist)))

            XTest[:, j] = image.reshape(-1)

        else:

            image = readPGM(datadir + '/' + testlist[j].rstrip('\n'))
            XTest[:, j] = image.reshape(-1)

    return XTrain, XTest, trainlist, testlist
