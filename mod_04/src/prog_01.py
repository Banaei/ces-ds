import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

# ****************************************
#              Functions
# ****************************************

def loss_function(thresholds, X, Y):
    return [lossForOne(t, X, Y) for t in thresholds]

def lossForOne(threshold, X, Y):
    estimated_output = [int(x>threshold) for x in X]
    delta = abs(np.subtract(estimated_output, Y)).T
    return np.dot(delta, np.ones(X.shape))


# number of tagged images
taggedImagesCount = 300

# indexes of images correspoindig to a scene change (the first image is 1 and the last 150)
sequenceChangeIndexes = np.asarray([12, 13, 17, 19, 24, 29, 32, 35, 37, 46, 63, 68, 69, 72, 74, 80, 84, 86, 97, 105, 114, 117, 122, 137, 141, 147, 157, 160, 165, 175, 177, 180, 183, 195, 198, 201, 205, 213, 218, 226, 229, 236, 237, 243, 251, 257, 263, 268, 269, 272, 274, 279, 284, 288, 296, 298])

# converting to a zero starting list (from 0 to 149)
sequenceChangeIndexes = sequenceChangeIndexes -1

# Creating a learning data array with 1 for indexes corresponding to a frame change
training_Y = np.zeros(taggedImagesCount)
training_Y[sequenceChangeIndexes]=1

# getting distances from the saved file
distances =  np.load("C:/Users/abanaei/Google Drive/work/FormationDS/Cours/Mod04_SignalProcessing/TP/distances-data.npy")

# Grabbing the first 150 images corresponding to the training data
training_X = distances[range(taggedImagesCount)]

plt.hist(training_X[training_Y==0])
plt.hist(training_X[training_Y==1])


[int(x>10000) for x in training_X]

thresholds = np.arange(8000,15000,100)
losses = loss_function(thresholds, training_X, training_Y)
plt.plot(thresholds, losses)


x_min = optimize.brent(lossForOne, (training_X, training_Y), full_output=True)

res = minimize_scalar(lossForOne, 0, args=(training_X, training_Y), bounds=(0, 50000), method='bounded')


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(training_X.T, training_Y)
y_pred = logreg.predict(training_Y)

logreg.score(y_pred, training_Y)


