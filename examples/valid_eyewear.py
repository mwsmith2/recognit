import os
import numpy as np
from recognit import *

base = os.path.dirname(__file__)
path = os.path.realpath(base + '/../data/faces')
flist = ['Nearest Neighbor', '3NN', 'gmm', 'LDA', 'SVM (rbf)']

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.set_params(seed=895, train=0.60, valid=0.15)
faces.get_labels()

res = ['0', '2', '4']
rate = np.zeros((3, 5))

for i, r in enumerate(res):

    res2 = res[:]
    res2.remove(r)

    faces.get_data(id=3, exclude={4: res2})
    faces.create_matrices()

    # Start principal component analysis.
    clf = pca.PCA(faces.xtrain)
    clf.find_principal_components()
    clf.transform()

    predictor = predict.Predictor(clf.xtransform, faces.ytrain)

    funclist = [predictor.distance, predictor.knn, predictor.gmm,
                predictor.lda, predictor.svm]

    for j, func in enumerate(funclist):

        success = 0

        for n, y in enumerate(faces.yvalid):

            s = clf.project(faces.xvalid[:, n])
            predictor.set_data(s)
            prediction = func()

            if prediction == y:

                success += 1.0

        rate[i, j] = (success / len(faces.yvalid))

    idx = rate[i, :].argmax()
    print "Most successful in predicting eyewear is", flist[idx]
    print "{0: .2f}% success for {1}".format(rate[i, :].max() * 100, r)
