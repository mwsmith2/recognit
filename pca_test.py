#pca_test.py

import analyze
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np

Xtrain, Xtest = analyze.createMatrices('faces/straight_open_4_faces.txt')

Evalue, Eface = analyze.PCA(Xtrain)

print len(Eface[0])

for i in range(10):
	plt.imshow(np.reshape(Eface[i], [30, 32]))
	plt.savefig('fig/eface' + str(i) + '_test.pdf')
	plt.close()

