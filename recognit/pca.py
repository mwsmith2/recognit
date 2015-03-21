"""pca.py: Contains a class for Principal Component Analysis."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
from scipy.linalg import eig, norm


class PCA(object):
    """Class that handles dimensionality reduction via princinpal 
    component anlysis."""

    def __init__(self, x):

        # Initialize x from data and calculate average.
        self.x = x
        self.d, self.N = x.shape
        self.mu = x.mean(axis=-1).reshape(self.d, 1)

    def find_principal_components(self, k=None, threshold=0.90):
        """Finds the principal components of the given data set.

        Parameters
        ----------

        k : integer
            
            Forces the number of relevant principal components to be k"

        threshold : float

            Gives a reference point for determining the number of 
            relevant principal components.  That number being the total
            number of components needed to explain 'threshold' of 
            of the total variance.

        Returns
        -------
        None

        """

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

        # Store only the most significant k eigenvectors.
        self.eigvec = eigvec[:, :self.k]
        self.eigval = eigval[:self.k]

    def project(self, s):
        """Decomposes the matrix 's' into weights of the relevant
        principal components.

        Parameters
        ----------
        s : array-like

            The image to be decomposed.

        Returns
        -------
        None

        """

        # Check shape and reshape if necessary.
        if s.shape != self.mu.shape:

            s = s.reshape(self.mu.shape)

        # Project s onto eigenvectors.
        return np.dot(self.eigvec.T, s - self.mu)

    def transform(self):
        """Projects all data into the reduced dimensionality space.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.xtransform = np.dot(self.eigvec.T, self.x)
