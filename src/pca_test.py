#pca_test.py

import analyze
import load
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

size = ['0', '2', '4']
t = np.linspace(0.5, 1, 40)

neig = np.zeros((len(size), len(t)))
rate = np.zeros((len(size), len(t)))

for i in range(len(size)):

    filename = 'faces/straight_open_' + size[i] + '_faces.txt'
    XTrain, XTest, trainlist, testlist = load.createMatrices(filename, sd=659, ratio=0.65)

    for j in range(len(t)):

        weightDict, eigvecs, meanface = analyze.characterizeFaces(XTrain, trainlist, threshold=t[j])

        success = 0
        fail = 0

        for imgname in testlist:

            image = load.readPGM('faces/' + imgname.rstrip('\n'))
            weight = analyze.calcWeight(image.reshape(-1), eigvecs, meanface)
            person = analyze.guessDistance(weightDict, weight)

            if person == imgname.split('_')[0]:

                success += 1

            else:

                fail += 1

        neig[i, j] = len(weight)
        rate[i, j] = 100 * success / float(success + fail)

plt.plot(t, neig[0, :], 'r', label="High resolution", linewidth=2.0)
plt.plot(t, neig[1, :], 'g', label="Med resolution", linewidth=2.0)
plt.plot(t, neig[2, :], 'b', label="Low resolution", linewidth=2.0)
plt.xlim((0.5, 1))
plt.xlabel('Threshold')
plt.ylabel('Number of Eigenvectors')
plt.legend(loc=0)

#plt.savefig("neig.pdf")
plt.close()

plt.plot(t, rate[0, :], 'r', label="High resolution", linewidth=2.0)
plt.plot(t, rate[1, :], 'g', label="Med resolution", linewidth=2.0)
plt.plot(t, rate[2, :], 'b', label="Low resolution", linewidth=2.0)
plt.xlim((0.5, 1))
plt.xlabel('Threshold')
plt.ylabel('Success Rate [%]')
plt.legend(loc=0)

#plt.savefig("rate.pdf")
plt.close()
