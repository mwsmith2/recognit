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

class Faces():

    def __init__(self, filename, sd=1234, ratio=0.6):

        self.XTrain, self.XTest, trainlist, testlist = createMatrices(filename, sd=sd, ratio=ratio) 

        self.YTrain = []
        for labels in trainlist:
            self.YTrain.append(labels[6:-5].split('_'))
            # 6:-5 cuts off faces/ prefix and .pgd suffix  

        self.YTest = []
        for labels in testlist:
            self.YTest.append(labels[6:-5].split('_'))

        self.XTrainCut = self.XTrain
        self.XTestCut  = self.XTest
        self.YTrainCut = list(self.YTrain)
        self.YTestCut  = list(self.YTest)

        # A few more list to hold our cuts
        self.blacklist = [] # Use to cut from data
        self.whitelist = []
        for s in self.YTrain[0]:
            self.whitelist.append("")

        # Store the current number of samples
        self.nTestPoints = self.XTestCut.shape[1]
        self.nTrainPoints = self.XTrainCut.shape[1]
        self.upToDate = True

    def flag(self, badLabel):

        if isinstance(badLabel, list):
            for label in badLabel:
                self.blacklist.append(label)
        else:
            self.blacklist.append(badLabel)

        self.upToDate = False

    def clearFlags(self):

        self.blacklist = []

        self.upToDate  = False

    def force(self, labelID, label=""):

        try: 
            self.whitelist[labelID] = label

        except IndexError:
            print "No such label ID."
            pass

        self.upToDate = False

    def clearForce(self):

        self.whitelist = []
        for s in self.YTrain[0]:
            self.whitelist.append('')

        self.upToDate = False

    def build(self):

        # Check if needs to be rebuilt
        if self.upToDate:
            return

        # Rebuild the data and labels with cuts
        cutIdx = 0
        oldIdx = 0
        for y in self.YTrain:

            flagged = False

            for label in y:
                if label in self.blacklist:
                    flagged = True

            for w in self.whitelist:
                if (w != '' and w not in y):
                    flagged = True

            if flagged:
                oldIdx += 1
                continue

            print cutIdx, oldIdx

            self.XTrainCut[:, cutIdx] = self.XTrain[:, oldIdx]
            self.YTrainCut[cutIdx] = y

            cutIdx += 1
            oldIdx += 1

        self.nTrainPoints = cutIdx

        cutIdx = 0
        oldIdx = 0
        for y in self.YTest:

            flagged = False

            for label in y:
                if label in self.blacklist:
                    flagged = True


            for w in self.whitelist:
                if (w != '' and w not in y):
                    flagged = True

            if flagged:
                oldIdx += 1
                continue

            print cutIdx, oldIdx

            self.XTestCut[:, cutIdx] = self.XTest[:, oldIdx]
            self.YTestCut[cutIdx] = y

            cutIdx += 1
            oldIdx += 1

        self.nTestPoints = cutIdx
        self.upToDate = True

    def getTrain(self, labelID):

        self.build()
        YTrain = []
        for y in self.YTrainCut:
            YTrain.append(y[labelID])

        return self.XTrainCut[:, :self.nTrainPoints], YTrain[:self.nTrainPoints]

    def getTest(self, labelID):

        self.build()
        YTest = []
        for y in self.YTestCut:
            YTest.append(y[labelID])

        return self.XTestCut[:, :self.nTestPoints], YTest[:self.nTestPoints]







