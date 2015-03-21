"""predict.py: Contains a class for classifiers."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"
__status__ = 'Development'

from sklearn.lda import LDA
from recognit.pca import PCA


class Predictor(object):
    """Class for different classifiers to predict labels."""

    def __init__(self, clf, dim_reduce='pca'):

        # Load training data and label.
        self.dim_reduce = 'pca'
        self.clf = clf
        self.k = None
        self.threshold = 0.90

    def train(self, x=None, y=None):
        """Train the dataset with the given labels and clf,
        if no data is given use the last dataset and return classification
        results.

        Parameters
        ----------
        x : array, shape(n_features, n_samples)

            the training data, flattened images

        y : array, shape(n_samples)

            the training labels, some trait, i.e. 'name', 'orientation', etc.

        Returns
        -------
        None

        """
        refit = False

        if x is not None:
            self.x = x
            refit = True

        if y is not None:
            self.y = y
            refit = True

        if refit is True:

            if self.dim_reduce == 'pca':

                dimred = PCA(x)
                dimred.find_principal_components(self.k, self.threshold)
                self.x = dimred.transform(x)

            elif self.dim_reduce == 'lda':

                dimred = LDA(self.k)
                dimred.fit(self.X, self.y)
                self.x = dimred.transform(x)

        self.clf.fit(self.x, self.y)

    def predict(self, x):
        """Predict classifier for the given dataset.

        Parameters
        ----------
        x : array, shape(n_features, n_samples)

            the test data, flattened images

        Returns
        -------
        ypredict : array, shape(n_samples)

            the predicted labels for some trait

        """
        x = self.dimred.transform(x)
        return self.clf.predict(x)

    def test(self, xtest, ytest):
        """Test the fidelity of predictions made by the training set.

        Parameters
        ----------
        x : array, shape(n_features, n_samples)

            the test data, flattened images

        y : array, shape(n_samples)

            the test labels for some trait

        Returns
        -------
        success_rate : float

            fraction of successful predictions in test labels

        """
        xtest = self.dimred.transform(xtest)
        ypredict = self.clf.predict(xtest)

        return (ytest == ypredict).sum() / ytest.shape[0]
