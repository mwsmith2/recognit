"""analysis.py: Contains classes for analysis of data."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
from scipy.linalg import eig, norm
from scipy.spatial.distance import cdist


class PCA:

    def __init__(self, x):

        # Initialize x from data and calculate average.
        self.x = x
        self.d, self.N = x.shape
        self.mu = x.mean(axis=-1).reshape(self.d, 1)

    def findPrincipalComponents(self, k=None, threshold=0.90):

        # Normalize x by subtracting average.
        self.x -= self.mu

        # Get eigenvalues and vectors of cov(x.T * x).
        cov = np.dot(self.x.T, self.x)
        eigval, eigvec = eig(cov)

        # Convert these to eigenvalues of the cov(X).
        eigvec = np.dot(self.x, eigvec)
        # Sort them in descending order.
        eigvec = eigvec[:, eigval.argsort()][:, ::-1]
        eigval.sort()
        eigval = eigval[::-1]

        # Select the first k ones, defined by k or threshold.
        if k is None:

            eignorm = eigval.cumsum() / eigval.sum()
            self.k = np.abs(eignorm - threshold).argmin()

        else:

            self.k = k

        # Normalize the eigenvectors.
        for i in range(self.k):

            eigvec[:, i] = eigvec[:, i] / norm(eigvec[:, i])

        # Return only the most significant k eigenvectors.
        self.eigvec = eigvec[:, :self.k]
        self.eigval = eigval[:self.k]

    def project(self, s):

        # Check shape and reshape if necessary.
        if len(s.shape) == 1:

            s = s.reshape(len(s), 1)

        # Project s onto eigenvectors.
        return np.dot(self.eigvec.T, s - self.mu)

    def createDatabase(self, y):

        # Create a database given list of features.
        self.y = np.array(y)
        self.weights = np.dot(self.eigvec.T, self.x)

    def predict(self, s, method='mahalanobis'):

        # Calculate weights for given vector.
        weight = self.project(s)

        # Predict by given method.
        if (method == 'mahalanobis'):

            distance = cdist(self.weights.T, weight.T, 'mahalanobis')
            prediction = self.y[distance.argmin()].tolist()

        return prediction


class LDA():

    def __init__(self, x, c):

        # Initialize x from data and calculate average.
        self.x = x
        self.d, self.N = x.shape
        self.mu = x.mean(axis=-1).reshape(self.d, 1)

        # Initialize classes and class averages.
        self.c = np.array(c)
        self.ulabels = np.unique(c)
        self.muc = np.empty((self.d, len(self.ulabels)))

        for i in range(len(self.ulabels)):

            self.muc[:, i] = x[:, c == self.ulabels[i]].mean(axis=-1)

    def findComponents(self, pcavec, k=0):

        if (k <= 0 or k > (len(self.ulabels) - 1)):

            k = len(self.ulabels) - 1

        self.k = k

        sw = np.zeros((self.d, self.d))
        sb = np.zeros((self.d, self.d))

        for i in range(len(self.ulabels)):

            cls = self.x[:, self.c == self.ulabels[i]]
            mcls = self.muc[:, i].reshape(self.d, 1)

            sw += np.dot((cls - mcls), (cls - mcls).T)
            sb += self.N * np.dot((mcls - self.mu), (mcls - self.mu).T)

        eigval, eigvec = eig(np.dot(np.linalg.inv(sw), sb))

        eigvec = eigvec[:, eigval.argsort()][:, ::-1]
        eigval.sort()
        eigval = eigval[::-1]

        self.eigvec = np.dot(pcavec, eigvec[:, :self.k]).real
        self.eigval = eigval[:self.k].real

    def project(self, s):

        # Check shape and reshape if necessary.
        if len(s.shape) == 1:

            s = s.reshape(len(s), 1)

        # Project s onto eigenvectors.
        weight = np.dot(self.eigvec.T, s - self.mufaces)

        return weight

    def createDatabase(self, x):

        # Create a database of faces.
        self.faces = x
        self.mufaces = self.faces.mean(axis=-1).reshape(len(x), 1)
        self.weights = np.dot(self.eigvec.T, self.faces)

    def predict(self, s, method='mahalanobis'):

        # Calculate weights for given image.
        weight = self.project(s)

        # Predict by given method.
        if (method == 'mahalanobis'):

            distance = cdist(self.weights.T, weight.T, 'mahalanobis')
            prediction = self.c[distance.argmin()].tolist()

        return prediction
