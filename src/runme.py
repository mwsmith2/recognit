import load
import pca
import predict

path = 'C:/Users/ukara_000/SkyDrive/UW/Courses/CSE546/Project/data/faces'

# Load faces from the path.
faces = load.Faces(path, 'pgm')
faces.setParams(seed=1234)
faces.getLabels()
faces.getData(ID=0, exclude={4: ['0', '2']})
faces.createMatrices()

# Start principal component analysis.
clf = pca.PCA(faces.XTrain)
clf.findPrincipalComponents()
clf.transform()

predictor = predict.Predictor(clf.xtransform, faces.YTrain)

success = []
fail = []

for i, y in enumerate(faces.YTest):

    predictor.setData(faces.XTest[i, :])
    prediction = predictor.distance()

    if (prediction == y):

        success.append(prediction)

    else:

        fail.append(prediction)

rate = 100 * len(success) / float(len(success + fail))
print rate