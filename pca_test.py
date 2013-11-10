#pca_test.py

import analyze
import load
import numpy as np

filename = 'faces/straight_open_2_faces.txt'

XTrain, XTest, trainlist, testlist = analyze.createMatrices(filename)
weightDict, eigvecs = analyze.characterizeFaces(XTrain, trainlist)

success = 0
successlist = []

fail = 0
faillist = []

for imgname in testlist:

    image = load.readPGM('faces/' + imgname.rstrip('\n'))
    weight = analyze.calcWeight(image.reshape(-1), eigvecs)
    person = analyze.guessWho(weightDict, weight)

    if person == imgname.split('_')[0]:

        success += 1
        successlist.append(imgname)

    else:

        fail += 1
        faillist.append(imgname)