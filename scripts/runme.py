from methods import *
import numpy as np

path = '../data/faces'
rate = []

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.setParams(seed=895, train=0.60, valid=0.15)
faces.getLabels()

res = ['0', '2', '4']
rate = np.zeros((3, 4))

for i, r in enumerate(res):

    for j in range(4):

        res2 = res[:]
        res2.remove(r)

        faces.getData(ID=j, exclude={4: res2})
        faces.createMatrices()

        # Start principal component analysis.
        clf = pca.PCA(faces.XTrain)
        clf.findPrincipalComponents()
        clf.transform()

        predictor = predict.Predictor(clf.xtransform, faces.YTrain)

        funclist = [predictor.lda, predictor.kNN, predictor.GMM,
                    predictor.distance]

        success = 0

        for n, y in enumerate(faces.YTest):

            s = clf.project(faces.XTest[:, n])
            predictor.setData(s)
            prediction = funclist[j]()

            if prediction == y:

                success += 1.0

        rate[i, j] = success / len(faces.YTest)
        print "%{0: .2f} success in predicting feature {1} for resolution {2}.".format(rate[i, j] * 100, j, r)