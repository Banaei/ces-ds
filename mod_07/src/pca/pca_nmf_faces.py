# -*- coding: utf-8 -*-

# Authors: Vlad Niculae, Alexandre Gramfort, Slim Essid
# License: BSD

from time import time
from numpy.random import RandomState
import pylab as pl
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition

# -- Prepare data and define utility functions ---------------------------------

n_row, n_col = 2, 5
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0, dtype=np.float64)

print "Dataset consists of %d faces" % n_samples

def plot_gallery(title, images):
    pl.figure(figsize=(2. * n_col, 2.26 * n_row))
    pl.suptitle(title, size=16)
    for i, comp in enumerate(images):
        pl.subplot(n_row, n_col, i + 1)
        
        comp = comp.reshape(image_shape)
        vmax = comp.max()
        vmin = comp.min()
        dmy = np.nonzero(comp<0)
        if len(dmy[0])>0:
            yz, xz = dmy            
        comp[comp<0] = 0

        pl.imshow(comp, cmap=pl.cm.gray, vmax=vmax, vmin=vmin)
        #print "vmax: %f, vmin: %f" % (vmax, vmin)
        #print comp
        
        if len(dmy[0])>0:
            pl.plot( xz, yz, 'r,', hold=True)
            print len(dmy[0]), "negative-valued pixels"
                  
        pl.xticks(())
        pl.yticks(())
        
    pl.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)



    
    
# Plot a sample of the input data
plot_gallery("OriginalOlivetti faces", faces[:n_components])

# Plot a sample of the input data
plot_gallery("First centered Olivetti faces", faces_centered[:n_components])


# -- Decomposition methods -----------------------------------------------------

# List of the different estimators and whether to center the data

estimators = [
    ('pca', 'Eigenfaces - PCA',
     decomposition.PCA(n_components=n_components, whiten=True),
     True),

    ('nmf', 'Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init=None, tol=1e-6, 
                       sparseness=None, max_iter=1000), 
     False)
]

# -- Transform and classify ----------------------------------------------------

from sklearn.cross_validation import cross_val_score
from sklearn.lda import LDA

labels = dataset.target

training_data_size = 300

X = faces
X_ = faces_centered

for shortname, name, estimator, center in estimators:
#    if shortname != 'nmf': continue
    print "Extracting the top %d %s..." % (n_components, name)
    t0 = time()
 
    data = X
    if center:
        data = X_
        
    H = estimator.fit_transform(data) # Donnees reduites

    train_time = (time() - t0)
    print "done in %0.3fs" % train_time
 
    components_ = estimator.components_
    
    plot_gallery('%s - W : Train time %.1fs' % (name, train_time), components_[:n_components])
    plot_gallery('%s - Original Images' % (name), faces[:n_components])
    plot_gallery('%s - Reduced Images' % (name), np.dot(H,components_)[:n_components])

    X_training = H[0:training_data_size,:]
    y_training = labels[0:training_data_size]
    X_validation = H[training_data_size:,:]
    y_validation = labels[training_data_size:]

    lda = LDA()
    cv_score = cross_val_score(lda, X_training, y_training, cv=5)
    print "CV Score LDA on %s with %s training data on %s training data = %s" %(name, training_data_size, labels.size-training_data_size, np.mean(cv_score))
    lda.fit(X_validation, y_validation)
    val_score = lda.score(X_validation, y_validation)
    print "Score LDA on %s with %s training data on %s validation data = %s" %(name, training_data_size, labels.size-training_data_size, val_score)

def get_h_w_estimator(estimator, centered):
    X = faces
    X_ = faces_centered
    data = X
    if centered:
        data = X_
    return estimator.fit_transform(data), estimator.components_ 

estimator_pca = decomposition.PCA(n_components=n_components, whiten=True)
H_pca, W_pca = get_h_w_estimator(estimator_pca, True)

estimator_nmf = decomposition.NMF(n_components=n_components, init=None, tol=1e-6, sparseness=None, max_iter=1000)
H_nmf, W_nmf = get_h_w_estimator(estimator_nmf, False)
