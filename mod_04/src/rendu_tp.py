import numpy as np
import matplotlib.pyplot as plt
import rendu_tp_functions as f


# imagesDirectoryPath = "C:/data/SignalProcessing/images"
imagesDirectoryPath = "d:/Alireza/data/images"


# number of tagged images
taggedImagesCount = 300

# indexes of images correspoindig to a scene change (the first image is 1 and the last 150)
sequenceChangeIndexes = np.asarray([12, 13, 17, 19, 24, 29, 32, 35, 37, 46, 63, 68, 69, 72, 74, 80, 84, 86, 97, 105, 114, 117, 122, 137, 141, 147, 157, 160, 165, 175, 177, 180, 183, 195, 198, 201, 205, 213, 218, 226, 229, 236, 237, 243, 251, 257, 263, 268, 269, 272, 274, 279, 284, 288, 296, 298])

# converting to a zero starting list (from 0 to 149)
sequenceChangeIndexes = sequenceChangeIndexes -1

# Creating a learning data array with 1 for indexes corresponding to a frame change
training_Y = np.zeros(taggedImagesCount)
training_Y[sequenceChangeIndexes]=1


# getting data from images on disk
# Getting distances for all images (1124 in total)
# The following 2 lines are commented because distances have already been saved on the disk
distances = f.calculAllDistances(imagesDirectoryPath)
# saveNpArray("distances-data", distances)

imageNames = f.filesNamesInDirectory(imagesDirectoryPath)

# Alternatively getting distances from the saved file
# distances =  np.load("C:/Users/abanaei/Google Drive/work/FormationDS/Cours/Mod04_SignalProcessing/TP/distances-data.npy")


# Grabbing the first 150 images corresponding to the training data
training_X = distances[range(taggedImagesCount)]


# Histogramme des distances pour les deux classes de Y
plt.hist(training_X[training_Y==0])
plt.hist(training_X[training_Y==1])


thresholds = np.arange(0,50000,100)
losses = f.loss_function(thresholds, training_X, training_Y)
plt.plot(thresholds, losses)


# ***************************************************************
#           Classification par la regression logistique
# ***************************************************************
from sklearn.linear_model import LogisticRegression

training_X = training_X.reshape((training_X.shape[0],1))
training_Y = training_Y.reshape((training_Y.shape[0],1))

logreg = LogisticRegression()
logreg.fit(training_X, training_Y)
y_pred = logreg.predict(training_X)
y_pred = y_pred.reshape((y_pred.shape[0],1))
logreg.score(y_pred, training_Y)

r = (np.sum((training_Y==y_pred).astype(int))/float(training_Y.shape[0]))
print "Precision de la regression logistique : %s" % (r)

# Calcul des changements de scene pour toutes les images
X = distances.reshape((distances.shape[0],1))
y = logreg.predict(X)


# creation du fichier html
f.createHtmlForImages('lr_output.html', imageNames, y)


# ***************************************************************
#           Classification par minimisation de la finction de cout
# ***************************************************************

#  
from scipy.optimize import minimize_scalar

training_X = np.squeeze(training_X)
training_Y = np.squeeze(training_Y)

res = minimize_scalar(f.lossForOne, 0, args=(training_X, training_Y), bounds=(0, 50000), method='bounded')
threshold = res['x']

y_pred = (training_X>threshold).astype(int)
r = (np.sum((training_Y==y_pred).astype(int))/float(training_Y.shape[0]))

print "Precision de la minimisation : %s" % (r)

y = (distances>threshold).astype(int)
f.createHtmlForImages('minimize_output.html', imageNames, y)


