#pca_test.py

import analyze
import load

filename = 'faces/straight_open_4_faces.txt'

XTrain, XTest, trainlist, testlist = load.createMatrices(filename)
weightDict, eigvecs, meanface = analyze.characterizeFaces(XTrain, trainlist)

success = 0
successlist = []
successdistance = []

fail = 0
faillist = []
faildistance = []

for imgname in testlist:

    image = load.readPGM('faces/' + imgname.rstrip('\n'))
    weight = analyze.calcWeight(image.reshape(-1), eigvecs, meanface)
    person, distance = analyze.guessLabel(weightDict, weight)

    if person == imgname.split('_')[0]:

        success += 1
        successlist.append(imgname)
        successdistance.append(distance)

    else:

        fail += 1
        faillist.append(imgname)
        faildistance.append(distance)