"""pca.py: Contains a class for Principal Component Analysis (PCA)."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"

import numpy as np
from scipy.linalg import eig, norm
from scipy.spatial.distance import cdist


class PCA:
    
    def __init__(self, x):        

        # Initialize x and average face.
        self.x = x
        self.d, self.N = x.shape 
        self.mu = x.mean(axis=-1).reshape(self.d, 1)
        
    def findPrincipalComponents(self, threshold=0.90):
        
        # Normalize x by subtracting average face.
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
        
        # Select the first k ones, which are responsible of 95% variance.
        eignorm = eigval.cumsum() / eigval.sum()
        self.k = np.abs(eignorm - threshold).argmin() 

        # Normalize the eigenfaces.
        for i in range(self.k):

            eigvec[:, i] = eigvec[:, i] / norm(eigvec[:, i])
            
        self.eigvec = eigvec[:, :self.k]
        self.eigval = eigval[:self.k]
        
    def project(self, s):
        
        # Check shape and reshape if necessary.
        if len(s.shape) == 1:
            
            s = s.reshape(len(s), 1)        
        
        # Project s onto eigenfaces.
        weight = np.dot(self.eigvec.T, s - self.mu)
            
        return weight
            
    def createDatabase(self, y):
        
        # Create a database of faces given list of features.
        self.y = np.array(y)
        self.weights = np.dot(self.eigvec.T, self.x)
        
    def predict(self, s, method='1NN'):
        
        # Calculate weights for given image.
        weight = self.project(s)
            
        # Predict by given method.
        if (method == '1NN'):
            
            distance = cdist(self.weights.T, weight.T, 'mahalanobis')
            prediction = self.y[distance.argmin()].tolist()
            
        return prediction

class LDA():
    
    def __init__(self, x, c):
        
        # Initialize x and average face.
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
        
        # Project s onto eigenfaces.
        weight = np.dot(self.eigvec.T, s - self.mufaces)
            
        return weight
            
    def createDatabase(self, x):
        
        # Create a database of faces.
        self.faces = x
        self.mufaces = self.faces.mean(axis=-1).reshape(len(x), 1)
        self.weights = np.dot(self.eigvec.T, self.faces)            
            
    def predict(self, s, method='1NN'):
    
        # Calculate weights for given image.
        weight = self.project(s)
        
        # Predict by given method.
        if (method == '1NN'):
        
            distance = cdist(self.weights.T, weight.T, 'mahalanobis')
            prediction = self.c[distance.argmin()].tolist()
        
        return prediction
        

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
        self.rk = np.ones(self.N) # This is the sum of each r column
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
            self.mu[k]  = (self.r[:, k] * w).sum(axis=1) / self.rk[k]

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
            self.pi  = self.rk / self.N
            for k in range(self.N):
                self.mu[k]  = (self.r[:,k] * w).sum(axis=1) / self.rk[k]
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



    
        