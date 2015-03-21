import os
import time

from sklearn.decomposition import PCA

from recognit import *


base = os.path.dirname(__file__)
path = os.path.realpath(base + '/../data')

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.set_params(seed=895, train=0.60, valid=0.15)
faces.get_labels()

res = 0
n = 1
ntests = 4
trait = 0
nweights = 5 # plots all weights less against each other

prefix = 'res_' + str(res) + '_trait_' + str(trait) + '_'
reslist = ['0', '2', '4']
reslist.remove(str(res)) 

faces.get_data(id=trait, exclude={4: reslist})
faces.create_matrices()

# Time out principal component analysis.

print "Testing our PCA speed."

for i in range(ntests):

	t0 = time.time()

	clf = pca.PCA(faces.xtrain)
	clf.find_principal_components()
	clf.transform()

	n_comp = clf.k

	print "Round %i: %f s." % (i, t0 - time.time())


print "Testing scikit-learn PCA speed."

for i in range(ntests):

	t0 = time.time()
	clf = PCA(n_components=n_comp)
	clf.fit(faces.xtrain.T)
	clf.transform(faces.xtrain.T)

	print "Round %i: %f s." % (i, t0 - time.time())

print "Comparing results of both PCA classes."

# Our pca first.
clf = pca.PCA(faces.xtrain)
clf.find_principal_components()
clf.transform()
our_eigvecs = clf.eigvec

# Now sklearn pca.
clf = PCA(n_components=n_comp)
clf.fit(faces.xtrain.T)
clf.transform(faces.xtrain.T)
skl_eigvecs = clf.components_.T

print our_eigvecs.shape, skl_eigvecs.shape

E = []
for i in xrange(min(len(our_eigvecs), 16)):
	e = our_eigvecs[:,i].reshape([120/n, 128/n]) 
	E.append(visual.normalize(e, 0, 255))

fn = "our_eigenfaces.pdf"
visual.plot_faces(title="Eigenfaces", images=E, sptitle=" Eigenface", filename=fn)

E = []
for i in xrange(min(len(skl_eigvecs), 16)):
	e = skl_eigvecs[:,i].reshape([120/n, 128/n]) 
	E.append(visual.normalize(e, 0, 255))

fn = "skl_eigenfaces.pdf"
visual.plot_faces(title="Eigenfaces", images=E, sptitle=" Eigenface", filename=fn)

E = []
for i in xrange(min(len(skl_eigvecs), 16)):
	e = (our_eigvecs[:, i] - skl_eigvecs[:,i]).reshape([120/n, 128/n]) 
	E.append(visual.normalize(e, 0, 255))

fn = "diff_eigenfaces.pdf"
visual.plot_faces(title="Eigenfaces", images=E, sptitle=" Eigenface", filename=fn)
