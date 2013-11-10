"""load.py: Contains functions to load PGM file types."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
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
