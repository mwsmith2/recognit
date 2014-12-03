from methods import *
import numpy as np

path = '../data/faces'
flist = ['Nearest Neighbor', '3NN', 'GMM', 'LDA', 'SVM (rbf)']

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.setParams(seed=895, train=0.60, valid=0.15)
faces.getLabels()

res = ['0', '2', '4']
rate = np.zeros((3, 5))

for i, r in enumerate(res):

    res2 = res[:]
    res2.remove(r)

    faces.getData(ID=2, exclude={4: res2})
    faces.createMatrices()

    # Start principal component analysis.
    clf = pca.PCA(faces.XTrain)
    clf.findPrincipalComponents()
    clf.transform()

    predictor = predict.Predictor(clf.xtransform, faces.YTrain)

    funclist = [predictor.distance, predictor.kNN, predictor.GMM,
                predictor.lda, predictor.svm]

    for j, func in enumerate(funclist):

        success = 0

        for n, y in enumerate(faces.YValid):

            s = clf.project(faces.XValid[:, n])
            predictor.setData(s)
            prediction = func()

            if prediction == y:

                success += 1.0

        rate[i, j] = (success / len(faces.YValid))

    idx = rate[i, :].argmax()
    print "Most successful in predicting mood is", flist[idx]
    print "{0: .2f}% success for {1}".format(rate[i, :].max() * 100, r)
