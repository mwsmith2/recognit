import load
import pca
import predict

# Loading stuff will come here.

clf = pca.PCA(XTrain)
clf.findPrincipalComponents()
clf.transform()

predictor = predict.Predictor(clf.xtransform, YTrain)

success = []
fail = []

for i, y in enumerate(YTest):

    predictor.setData(XTest[i, :])
    prediction = predictor.distance()

    if (prediction == y):

        success.append(prediction)

    else:

        fail.append(prediction)

rate = 100 * len(success) / float(len(success + fail)))
