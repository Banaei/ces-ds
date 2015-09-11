# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:45:27 2015

@author: Alireza
"""

# type here missing imports
import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans




def generate_data(n, centroids, sigma=1., random_state=42):
    """Generate sample data

    Parameters
    ----------
    n : int
        The number of samples
    centroids : array, shape=(k, p)
        The centroids
    sigma : float
        The standard deviation in each class

    Returns
    -------
    X : array, shape=(n, p)
        The samples
    y : array, shape=(n,)
        The labels
    """
    rng = np.random.RandomState(random_state)
    k, p = centroids.shape
    X = np.empty((n, p))
    y = np.empty(n)
    for i in range(k):
        X[i::k] = centroids[i] + sigma * rng.randn(len(X[i::k]), p)
        y[i::k] = i
    order = rng.permutation(n)
    X = X[order]
    y = y[order]
    return X, y


def compute_labels(X, centroids):
    """Compute labels

    Parameters
    ----------
    X : array, shape=(n, p)
        The samples

    Returns
    -------
    labels : array, shape=(n,)
        The labels of each sample
    """
    dist = spatial.distance.cdist(X, centroids, metric='euclidean')
    return dist.argmin(axis=1)


def compute_inertia_centroids(X, labels):
    """Compute inertia and centroids

    Parameters
    ----------
    X : array, shape=(n, p)
        The samples
    labels : array, shape=(n,)
        The labels of each sample

    Returns
    -------
    inertia: float
        The inertia
    centroids: array, shape=(k, p)
        The estimated centroids
    """
    # insert code here
    inertia = 0;
    centroids = []
    for i in np.unique(labels):
        X_l = X[labels==i]
        C_l = np.mean(X_l, axis=0)
        centroids.append(C_l)
        dist = X_l - C_l
        dist = dist**2;
        inertia += np.sum(dist)
#    print "Inertie=%s" %(inertia)
    return inertia, centroids


def kmeans(X, n_clusters, n_iter=100, tol=1e-7, random_state=42):
    """K-Means : Estimate position of centroids and labels

    Parameters
    ----------
    X : array, shape=(n, p)
        The samples
    n_clusters : int
        The desired number of clusters
    tol : float
        The tolerance to check convergence

    Returns
    -------
    centroids: array, shape=(k, p)
        The estimated centroids
    labels: array, shape=(n,)
        The estimated labels
    """
    # initialize centroids with random samples
    rng = np.random.RandomState(random_state)
    centroids = X[rng.permutation(len(X))[:n_clusters]]
    labels = compute_labels(X, centroids)
    old_inertia = np.inf
    for k in xrange(n_iter):
        inertia, centroids = compute_inertia_centroids(X, labels)
        if abs(inertia - old_inertia) < tol:
            print 'Converged !'
            break
        old_inertia = inertia
        labels = compute_labels(X, centroids)
        print inertia
    else:
        print 'Dit not converge...'
    # insert code here
    return centroids, labels


if __name__ == '__main__':
    n = 1000
    centroids = np.array([[0., 0.], [2., 2.], [0., 3.]])
    X, y = generate_data(n, centroids, random_state=42)

    centroids, labels = kmeans(X, n_clusters=3, random_state=2)
    # plotting code
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'or')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'ob')
    plt.plot(X[y == 2, 0], X[y == 2, 1], 'og')
    plt.title('Ground truth')

    plt.figure()
    plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'or')
    plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'ob')
    plt.plot(X[labels == 2, 0], X[labels == 2, 1], 'og')
    plt.title('Estimated')
    plt.show()

    k_means = KMeans(n_clusters=3)
    k_means.fit(X)
    labels = k_means.labels_
    plt.figure()
    plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'or')
    plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'ob')
    plt.plot(X[labels == 2, 0], X[labels == 2, 1], 'og')
    plt.title('SciKit Learn')
    plt.show()
    
n = 20
centroids = np.array([[0., 0.], [2., 2.], [0., 3.]])
X, y = generate_data(n, centroids, random_state=2)
l = compute_labels(X, centroids)
I, C = compute_inertia_centroids(X, l)