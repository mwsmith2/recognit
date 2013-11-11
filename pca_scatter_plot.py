#pca_test.py

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import analyze
import load
import numpy as np
from collections import defaultdict

filename = 'faces/straight_open_2_faces.txt'

XTrain, XTest, trainlist, testlist = load.createMatrices(filename, sd=659, ratio=0.65)
weightDict, eigvecs, meanface = analyze.characterizeFaces(XTrain, trainlist, threshold=0.91)

success = 0
successlist = []
successdistance = []

fail = 0
faillist = []
faildistance = []

scatterDict = defaultdict(list)
distanceDict = defaultdict(list)

for imgname in testlist:

    image = load.readPGM('faces/' + imgname.rstrip('\n'))
    weight = analyze.calcWeight(image.reshape(-1), eigvecs, meanface)
    person, distance = analyze.guessLabel(weightDict, weight)
    scatterDict[imgname.split('_')[0]].append(weight)
    distanceDict[imgname.split('_')[0]].append(distance)

    if person == imgname.split('_')[0]:

        success += 1
        successlist.append(imgname)
        successdistance.append(distance)

    else:

        fail += 1
        faillist.append(imgname)
        faildistance.append(distance)

print "succes = " + str(success)
print "fail = " + str(fail)

#for imgname in trainlist:

  #  image = load.readPGM('faces/' + imgname.rstrip('\n'))
  #  weight = analyze.calcWeight(image.reshape(-1), eigvecs, meanface)
  #  scatterDict[imgname.split('_')[0]].append(weight)
  #  distanceDict[imgname.split('_')[0]].append(distance)

colors = cm.hsv(np.linspace(0, 1, len(scatterDict)))
davg = np.mean(np.array(distanceDict.values()).reshape(-1)[0])
n1 = 0
n2 = 1

for name, color in zip(scatterDict, colors):

    w = np.array(scatterDict[name])
    dist = 50 * np.exp(0.004*(np.array(distanceDict[name]) - davg))
    plt.scatter(w[:,n1], w[:,n2], s=dist, c=color, alpha=0.5, label=name)

    if name in weightDict:
        wavg = weightDict[name]
        plt.scatter(wavg[n1], wavg[n2], c=color, alpha=0.5, marker='d')

plt.title('Facial Characterization')
plt.xlabel('Weight ' + str(n1 + 1))
plt.tick_params(axis='x',labelbottom='off')
plt.tick_params(axis='y',labelleft='off')
plt.xlim(plt.xlim()[0], plt.xlim()[1] + 500)
plt.ylabel('Weight ' + str(n2 + 1))
plt.legend(loc=5, fontsize='x-small')
plt.savefig('fig/scatter_by_name.pdf')
