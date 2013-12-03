import load
import pca
import predict
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

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

    faces.getData(ID=3, exclude={4: res2})
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
    print "Most successful in predicting person is", flist[idx]
    print "{0: .2f}% success for {1}".format(rate[i, :].max() * 100, r)

# Plotting

N = 5
ind = np.arange(N) + 0.125
width = 0.25

fig, ax = plt.subplots()

rects1 = ax.bar(ind, rate[0, :] * 100, width, color='#A6CEE3')
rects2 = ax.bar(ind + width, rate[1, :] * 100, width, color='#1F78B4')
rects3 = ax.bar(ind + 2 * width, rate[2, :] * 100, width, color='#B2DF8A')

ax.set_ylabel('Success Rate [%]', fontsize='xx-large')
ax.set_xticks(ind + 1.5 * width)
ax.set_xticklabels(('1NN', '3NN', 'GMM', 'LDA', 'SVM'), fontsize='xx-large')
ax.legend((rects1[0], rects2[0], rects3[0]),
          ('Hi-Res', 'Med-Res', 'Lo-Res'), fontsize='xx-large')
ax.tick_params(axis='both', which='major', labelsize='xx-large')
fig.set_size_inches(16, 12)
plt.savefig('../tex/fig/sunglasses.pdf', dpi=600)
