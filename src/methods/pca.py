"""pca.py: Contains a class for Principal Component Analysis."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
from scipy.linalg import eig, norm


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

        # Return only the mo=st significant k eigenvectors.
        self.eigvec = eigvec[:, :self.k]
        self.eigval = eigval[:self.k]

    def project(self, s):

        # Check shape and reshape if necessary.
        if len(s.shape) == 1:

            s = s.reshape(len(s), 1)

        # Project s onto eigenvectors.
        return np.dot(self.eigvec.T, s - self.mu)

    def transform(self):

        self.xtransform = np.dot(self.eigvec.T, self.x)
