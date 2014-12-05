import os

import matplotlib
matplotlib.use('PDF')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.lda import LDA
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

from rmf import load
from rmf import pca
from rmf import predict
from rmf import visual as vis

base = os.path.dirname(__file__)
path = os.path.realpath(base + '/../data/faces')
nweights = 5 #For plotting different omegas against eachother

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.set_params(seed=895, train=0.60, valid=0.15)
faces.get_labels()

for res in range(0, 6, 2):

	for trait in range(4):

		prefix = 'res_' + str(res) + '_trait_' + str(trait) + '_'
		reslist = ['0', '2', '4']
		reslist.remove(str(res))

		faces.get_data(id=trait, exclude={4: reslist})
		faces.create_matrices()

		# Start principal component analysis.
		clf = pca.PCA(faces.xtrain)
		clf.find_principal_components()
		clf.transform()

		predictor = predict.Predictor(clf.xtransform, faces.ytrain)

		success = []
		fail = []

		for i, y in enumerate(faces.ytest):

		    predictor.set_data(clf.project(faces.xtest[:, i]))
		    prediction = predictor.distance()

		    if (prediction == y):

		        success.append(prediction)

		    else:

		        fail.append(prediction)

		rate = 100 * len(success) / float(len(success + fail))
		print rate

		if trait == 0:
			# Plot the Eigenfaces first
			n = 2 ** (res / 2) # This guy resizes the images properly.
			E = []
			for i in xrange(min(len(clf.eigvec), 16)):
				e = clf.eigvec[:,i].reshape([120/n, 128/n]) 
				E.append(vis.normalize(e, 0, 255))
				
			fn = prefix + "eigenfaces.pdf"
			vis.plot_faces(title="Eigenfaces", images=E, sptitle=" Eigenface", filename=fn)


			# Plot our Fisherfaces
			lda = LDA()
			lda.fit(clf.xtransform.T, faces.ytrain)
			xtran = lda.transform(clf.xtransform.T)
			ff = np.dot(clf.eigvec[:,:xtran.shape[1]], xtran.T)

			E = []
			for i in xrange(min(len(ff), 16)):
				e = ff[:,i].reshape([120/n, 128/n]) 
				E.append(vis.normalize(e, 0, 255))

			fn = prefix + "fisherfaces.pdf"
			vis.plot_faces(title="Fisherfaces", images=E, sptitle=" Fisherface", filename=fn)

		# Plot the clusters given by first two weights
		for k in range(nweights):
			for j in range(k):

				faceweights = defaultdict(list)

				for i, y in enumerate(faces.ytest):
					s  = clf.project(faces.xtest[:, i])
					x1 = s[j]
					x2 = s[k]
					d  = np.min(cdist(clf.xtransform.T, s.T))
					faceweights[y].append([x1, x2, d])

				fn = prefix + "face_scatter_w" + str(j+1) + "_v_w" + str(k+1) + ".pdf"
				title = r'Principal Components: $\omega_' + str(j+1) + r'$ vs. $\omega_' + str(k+1) + r'$'
				vis.scatter_face(title, faceweights, x1=j+1, x2=k+1, filename=fn)

