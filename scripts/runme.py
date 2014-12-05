import os
import numpy as np
from rmf import *

base = os.path.dirname(__file__)
path = os.path.realpath(base + '/../data/faces')
rate = []

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.set_params(seed=895, train=0.60, valid=0.15)
faces.get_labels()

res = ['0', '2', '4']
rate = np.zeros((3, 4))

for i, r in enumerate(res):

    for j in range(4):

        res2 = res[:]
        res2.remove(r)

        faces.get_data(id=j, exclude={4: res2})
        faces.create_matrices()

        # Start principal component analysis.
        clf = pca.PCA(faces.xtrain)
        clf.find_principal_components()
        clf.transform()

        predictor = predict.Predictor(clf.xtransform, faces.ytrain)

        funclist = [predictor.lda, predictor.knn, predictor.gmm,
                    predictor.distance]

        success = 0

        for n, y in enumerate(faces.ytest):

            s = clf.project(faces.xtest[:, n])
            predictor.set_data(s)
            prediction = funclist[j]()

            if prediction == y:

                success += 1.0

        rate[i, j] = success / len(faces.ytest)
        print "%{0: .2f} success in predicting feature {1} for resolution {2}.".format(rate[i, j] * 100, j, r)