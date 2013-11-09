"""load.py: Contains functions to load PGM file types."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np


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

        image = decodeP5(f)
        f.close()

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


def decodeP5(f):
    """
    Decode P5 type PGM files.

    Parameters
    ----------
    f : file handler

            Handler of .pgm file.

    Returns
    -------
    image : array_like

            PGM file returned as numpy.ndarray with type int8 or int 16.

    """

    line = f.readline().split()  # This line contains the width and height.
    width = int(line[0])
    height = int(line[1])

    line = f.readline().split()  # This line contains the max value.
    maxval = int(line[0])

    if maxval < 256:

        pixelsize = 1  # Number of bytes.
        image = np.zeros([height, width], dtype='int8')

    else:

        pixelsize = 2
        image = np.zeros([height, width], dtype='int16')

    for n in range(height):

        buff = bytearray(f.read(width * pixelsize))

        for i in range(width):

            image[n, i] = int(buff[pixelsize * i])

            if pixelsize == 2:

                image[n, i] += (2 ** 8) * int(buff[pixelsize * i + 1])

    return image
