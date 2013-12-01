"""classification.py: Contains classes for classification of data."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np


class GMM():

    def __init__(self, w, nclusters=20, err=0.0001):

        self.findClusters(w, nclusters, err)

    def findClusters(self, w, nclusters, err):

        self.err = err
        self.w = w

        self.N = nclusters
        self.nweights = w.shape[0]
        self.nsamples = w.shape[1]

        # Initialize the responsibilities and gaussian params
        self.r = np.empty([self.nsamples, self.N])
        self.rk = np.ones(self.N)  # This is the sum of each r column
        self.pi = np.ones(self.N) / self.N
       # self.mu = np.zeros([self.N, self.nweights])
        self.mu = (2 * np.random.rand(self.N, self.nweights) - 1.0) * w.max()
        self.isig = np.empty([self.N, self.nweights, self.nweights])
        for k in range(self.N):
            self.isig[k] = np.diag(np.ones(self.nweights))

        # Initial E step
        for i in range(self.nsamples):

            p = np.empty(self.N)

            for k in range(self.N):

                p[k] = self.pi[k] * self.norm(self.w[:, i], self.mu[k], self.isig[k])

            self.r[i] = p / p.sum()

        self.rk = self.r.sum(axis=0)

        # Initial M step
        self.pi = self.rk / self.N
        for k in range(self.N):
            self.mu[k] = (self.r[:, k] * w).sum(axis=1) / self.rk[k]

            wsum = np.zeros([self.nweights, self.nweights])
            for i in range(self.nsamples):
                wsum += np.outer(w[:, i], w[:, i])

            self.isig[k] = np.linalg.inv(wsum / self.rk[k] - np.outer(self.mu[k], self.mu[k]))

        rold = np.zeros(self.r.shape)

        print self.rk.shape, self.rk.mean()
        print self.r.shape, self.r.mean()
        print self.mu.shape, self.mu.mean()
        print self.isig.shape, self.isig.mean()
        print self.pi.shape, self.pi.mean()

        count = 0
        while (np.abs(rold - self.r).max() > self.err):

            print "Iteration: " + str(count)
            count += 1
            print np.abs(rold - self.r).max()

            # Store old responsibilities to check tolerances
            rold = self.r.copy()

            # The E step
            for i in range(self.nsamples):

                p = np.empty(self.N)

                for k in range(self.N):

                    p[k] = self.pi[k] * self.norm(self.w[:, i], self.mu[k], self.isig[k])

                self.r[i] = p / p.sum()

            self.rk = self.r.sum(axis=0)

            # The M step
            self.pi = self.rk / self.N
            for k in range(self.N):
                self.mu[k] = (self.r[:,k] * w).sum(axis=1) / self.rk[k]
                wsum = np.zeros([self.nweights, self.nweights])
                for i in range(self.nsamples):
                    wsum += np.outer(w[:, i], w[:, i])

                self.isig[k] = np.linalg.inv(wsum / self.rk[k] - np.outer(self.mu[k], self.mu[k]))
                #self.isig[k] = np.linalg.inv(np.dot(self.r[:, k] * w, w.T) / self.rk[k] - np.outer(self.mu[k], self.mu[k]))

            print self.rk.shape, self.rk.mean(), self.rk.min(), self.rk.max()
            print self.r.shape, self.r.mean(), self.r.min(), self.r.max()
            print self.mu.shape, self.mu.mean(), self.mu.min(), self.mu.max()
            print self.pi.shape, self.pi.mean(), self.pi.min(), self.pi.max()
            print self.isig.shape, self.isig.mean(), self.isig.min(), self.isig.max()

    def norm(self, x, mu, isig):

        A = 1.0 / ((2 * np.pi) ** self.N / np.linalg.det(isig)) ** 0.5
        return A * np.exp(-0.5 * (x - mu).dot(isig.dot(x - mu)))
