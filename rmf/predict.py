"""predict.py: Contains a class for classifiers."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import gmm
from sklearn.svm import SVC
from sklearn.lda import LDA


class Predictor(object):
    """Class for different classifiers to predict labels."""

    def __init__(self, x, y):

        # Load training data and label.
        self.x = x.T
        self.y = y

    def set_data(self, s):

        # Load the data for label prediction.
        self.s = s.T

    def distance(self, method='mahalanobis'):
        """Nearest Neighbor algorithm using different distance metrics."""

        distance = cdist(self.x, self.s, method)
        prediction = self.y[distance.argmin()]

        return prediction

    def knn(self, k=3, weights='distance'):
        """knn algorithm with weighting option."""

        knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        knn.fit(self.x, self.y)
        prediction = knn.predict(self.s)[0]

        return prediction

    def gmm(self):
        """Gaussian Mixture Models algorithm."""

        # Use semi-supervised Gaussian mixtures.
        unique = np.unique(self.y)

        # Initialize means for different classes.
        means = np.empty((len(unique), self.x.shape[1]))

        for i in range(len(unique)):

            means[i, :] = self.x[self.y == unique[i]].mean(axis=0)

        gmm = gmm(n_components=len(unique), init_params='wc')
        gmm.means_ = means
        gmm.fit(self.x)
        prediction = unique[gmm.predict(self.s)][0]

        return prediction

    def svm(self, kernel='rbf', degree=3, gamma=0.0, C=100.0):
        """Support vector machines"""

        svm = SVC(kernel=kernel, degree=degree, gamma=gamma, C=C)
        svm.fit(self.x, self.y)
        prediction = svm.predict(self.s)[0]

        return prediction

    def lda(self):
        """Fisher's linear Discriminant Analysis"""

        lda = LDA()
        lda.fit(self.x, self.y)
        prediction = lda.predict(self.s)[0]

        return prediction
