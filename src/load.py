"""load.py: Contains functions to load PGM file types and create datasets."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
import os
import random
from matplotlib.image import imread


class Faces:

    def __init__(self, path, ext):

        self.path = path
        self.ext = ext
        self.filelist = []

        for f in os.listdir(self.path):

            if (f.endswith(self.ext) and
                    os.path.isfile(os.path.join(self.path, f))):

                self.filelist.append(f)

    def setParams(self, seed=None, train=0.5, valid=0.25):

        random.seed(seed)
        self.trr = train
        self.var = valid

    def getLabels(self, delimiter='_'):

        self.labels = []
        self.nlabels = len(self.filelist[0][:-4].split(delimiter))

        for f in self.filelist:

            self.labels.append(dict(enumerate(f[:-4].split(delimiter))))

        for i in range(self.nlabels):

            unique = set([label[i] for label in self.labels])
            print "ID = {0}: ".format(i), unique

    def getData(self, ID=0, exclude={0: None}):

        self.ID = ID
        self.exclude = exclude

        self.datalist = self.filelist[:]
        self.labellist = self.labels[:]

        for f, label in zip(self.filelist, self.labels):

            for key in self.exclude.keys():

                for feature in self.exclude[key]:

                    if (label[key] == feature):

                        try:

                            self.datalist.remove(f)
                            self.labellist.remove(label)

                        except ValueError:

                            pass

        self.fileload = [os.path.join(self.path, f) for f in self.datalist]
        self.y = [label[self.ID] for label in self.labellist]

    def loadMatrix(self, filelist):

        for i, f in enumerate(filelist):

            if i == 0:

                image = readPGM(f)

                height = image.shape[0]
                width = image.shape[1]

                X = np.empty((height * width, len(filelist)))

                X[:, i] = image.reshape(-1)

            else:

                image = readPGM(f)
                X[:, i] = image.reshape(-1)

        return X

    def createMatrices(self):

        randomizedlist = zip(self.fileload, self.y)
        random.shuffle(randomizedlist)

        nelements = len(randomizedlist)
        ntrain = int(self.trr * nelements)
        nvalid = int(self.var * nelements) + ntrain

        trainfilelist, YTrain = zip(*randomizedlist[:ntrain])
        validfilelist, YValid = zip(*randomizedlist[ntrain:nvalid])
        testfilelist, YTest = zip(*randomizedlist[nvalid:])

        self.XTrain = self.loadMatrix(trainfilelist)
        self.XValid = self.loadMatrix(validfilelist)
        self.XTest = self.loadMatrix(testfilelist)

        self.YTrain = list(YTrain)
        self.YValid = list(YValid)
        self.YTest = list(YTest)


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
