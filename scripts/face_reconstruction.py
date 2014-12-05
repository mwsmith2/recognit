import os

import matplotlib.cm as cm
from sklearn.lda import LDA
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from rmf import load
from rmf import pca
from rmf import predict
from rmf import visual as vis

base = os.path.dirname(__file__)
path = os.path.realpath(base + '/../data/faces')

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.set_params(seed=895, train=0.60, valid=0.15)
faces.get_labels()

res = 0
trait = 0
nweights = 5 # plots all weights less against each other

prefix = 'res_' + str(res) + '_trait_' + str(trait) + '_'
reslist = ['0', '2', '4']
reslist.remove(str(res)) 

faces.get_data(id=trait, exclude={4: reslist})
faces.create_matrices()

# Start principal component analysis.
clf = pca.PCA(faces.xtrain)
clf.find_principal_components()
clf.transform()

plt.imshow(faces.xtest[:, 0].reshape([120, 128]))
plt.savefig('original_face.pdf')
recon_weights = clf.project(faces.xtest[:, 0])

for i in range(clf.k+1):
	plt.close()
	X = np.dot(clf.eigvec[:, :i], recon_weights[:i]).sum(axis=1).reshape([120, 128])
	plt.imshow(X + clf.mu.reshape([120, 128]))
	if i % 10 == 0:
		plt.savefig('recon_' + str(i) + '.pdf')
 
