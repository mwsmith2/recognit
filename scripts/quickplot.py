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

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.setParams(seed=895, train=0.60, valid=0.15)
faces.getLabels()

faces.getData(ID=0, exclude={4: ['2', '4']})
faces.createMatrices()

# Start principal component analysis.
clf = pca.PCA(faces.XTrain)
clf.findPrincipalComponents()
clf.transform()

res = 0
trait = 0

d = []
x1 = []
x2 = []

for i in range(len(faces.YValid)):
	s  = clf.project(faces.XValid[:, i])
	d.append(np.min(cdist(clf.xtransform.T, s.T, "mahalanobis")))
	x1.append(s[0])
	x2.append(s[1])

for trait in range(4):
	prefix = 'res_' + str(res) + '_trait_' + str(trait) + '_'
	faces.setParams(seed=895, train=0.60, valid=0.15)
	faces.getData(ID=trait, exclude={4: ['2', '4']})
	faces.createMatrices()
	faceweights = defaultdict(list)
	
	for i, y in enumerate(faces.YValid):
		faceweights[y].append([x1[i], x2[i], d[i]])

	fn = prefix + "face_scatter_w" + str(1) + "_v_w" + str(2) + ".pdf"
	title = r'Principal Components: $\omega_1$ vs. $\omega_2$'
	vis.scatterFace(title, faceweights, filename=fn)

