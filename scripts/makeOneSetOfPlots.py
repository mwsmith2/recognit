import os

import matplotlib.cm as cm
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

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.setParams(seed=895, train=0.60, valid=0.15)
faces.getLabels()

res = 0
trait = 0
nweights = 5 # plots all weights less against each other

prefix = 'res_' + str(res) + '_trait_' + str(trait) + '_'
reslist = ['0', '2', '4']
reslist.remove(str(res))

faces.getData(ID=trait, exclude={4: reslist})
faces.createMatrices()

# Start principal component analysis.
clf = pca.PCA(faces.XTrain)
clf.findPrincipalComponents()
clf.transform()

predictor = predict.Predictor(clf.xtransform, faces.YTrain)

success = []
fail = []

for i, y in enumerate(faces.YTest):

    predictor.setData(clf.project(faces.XTest[:, i]))
    prediction = predictor.distance()

    if (prediction == y):

        success.append(prediction)

    else:

        fail.append(prediction)

rate = 100 * len(success) / float(len(success + fail))
print rate

# Plot the Eigenfaces first
n = 2 ** (res / 2) # This guy resizes the images properly.
E = []
for i in xrange(min(len(clf.eigvec), 18)):
	e = clf.eigvec[:,i].reshape([120/n, 128/n]) 
	E.append(vis.normalize(e, 0, 255))
	
fn = prefix + "eigenfaces.pdf"
vis.plotFaces(title="Eigenfaces", images=E, sptitle=" Eigenface", filename=fn)


# Plot our Fisherfaces
lda = LDA()
lda.fit(clf.xtransform.T, faces.YTrain)
xtran = lda.transform(clf.xtransform.T)
ff = np.dot(clf.eigvec[:,:xtran.shape[1]], xtran.T)

E = []
for i in xrange(min(len(ff), 18)):
	e = ff[:,i].reshape([120/n, 128/n]) 
	E.append(vis.normalize(e, 0, 255))

fn = prefix + "fisherfaces.pdf"
vis.plotFaces(title="Fisherfaces", images=E, sptitle=" Fisherface", filename=fn)

# Plot the clusters given by first two weights
for k in range(nweights):
	for j in range(k):

		faceweights = defaultdict(list)

		for i, y in enumerate(faces.YTest):
			s  = clf.project(faces.XTest[:, i])
			x1 = s[j]
			x2 = s[k]
			d  = np.min(cdist(clf.xtransform.T, s.T))
			faceweights[y].append([x1, x2, d])

		fn = prefix + "face_scatter_w" + str(j+1) + "_v_w" + str(k+1) + ".pdf"
		title = r'Principal Components: $\omega_' + str(j+1) + r'$ vs. $\omega_' + str(k+1) + r'$'
		vis.scatterFace(title, faceweights, x1=j+1, x2=k+1, filename=fn)

