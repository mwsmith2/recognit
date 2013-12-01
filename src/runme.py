import load
from pca import PCA, LDA


filename = 'data/straight_open_0_faces.txt'
XTrain, XTest, trainlist, testlist = load.createMatrices(filename, sd=659, ratio=0.65)

clf = PCA(XTrain)
clf.findPrincipalComponents()
clf.createDatabase(trainlist)

success = 0
fail = 0

for imgname in testlist:

    image = load.readPGM('data/faces/' + imgname.rstrip('\n'))
    person = clf.predict(image.reshape(-1))

    if person.split('_')[0] == imgname.split('_')[0]:

        success += 1

    else:

        fail += 1

    neig = clf.k
    rate = 100 * success / float(success + fail)

c = []
for string in trainlist:

    c.append(string.split('_')[0])

clLDA = LDA(clf.weights, c)
clLDA.findComponents(clf.eigvec)
clLDA.createDatabase(clf.x)

successLDA = 0
failLDA = 0

for imgname in testlist:

    image = load.readPGM('faces/' + imgname.rstrip('\n'))
    person = clLDA.predict(image.reshape(-1))

    if person == imgname.split('_')[0]:

        successLDA += 1

    else:

        failLDA += 1

    neig = clf.k
    rate = 100 * successLDA / float(successLDA + failLDA)
