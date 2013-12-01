import load
from analysis import PCA


filename = '../data/straight_open_0_faces.txt'
XTrain, XTest, trainlist, testlist = load.createMatrices(filename, sd=659, ratio=0.65)

clf = PCA(XTrain)
clf.findPrincipalComponents()
clf.createDatabase(trainlist)

methods = ['mahalanobis', '3NN', 'GMM', 'SVM', 'LDA']
rate = []

for mth in methods:

    success = 0
    fail = 0

    for imgname in testlist:

        image = load.readPGM('../data/' + imgname.rstrip('\n'))
        person = clf.predict(image.reshape(-1), method=mth)

        if person == imgname.split('_')[0]:

            success += 1

        else:

            fail += 1

    rate.append(100 * success / float(success + fail))

print rate