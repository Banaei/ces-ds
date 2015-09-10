
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from scipy import optimize


# ******************************************
#                 Functions
# ******************************************

def calculHistogram(imageName):
    im=cv2.imread(imageName)
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv],[0],None,[180],[0,180])

def calculDistance( sign1, sign2, flag=1 ):
    #vérification des dimensions
    L1 = len(sign1)
    L2 = len(sign2)
    if (L1 == L2 ):
        #calcul de la distance ou de la similarité
        if flag==1:
            d = 0;
            for i in range(L1):
                d=d+ (sign1[i]-sign2[i])**2
            d = np.sqrt( d )
        elif flag==2:
            d = 0;
            for i in range(L1):
                d=d+np.abs(sign1[i]-sign2[i])
        else:
            d = 0;
            for i in range(L1):
                d=d+sign1[i]/sign2[i]*np.log(sign1[i]/sign2[i])
        return d
    else:
        return np.inf

def filesNamesInDirectory(directory):
    for root, dirs, files in os.walk(directory):
        fileNames = [];
        for name in files:
            if name.endswith((".jpg")) :
                fileNames.append(name)
        return fileNames
        

def calculAllDistances(imagesDirectoryPath):   
    fileNames = filesNamesInDirectory(imagesDirectoryPath)
    distances = np.zeros(len(fileNames))
    firstLoop = True
    i=0
    for fileName in fileNames:
        filePath = imagesDirectoryPath+"/"+fileName
        if firstLoop:
            h1 = calculHistogram(filePath)
            firstLoop = False
        h = calculHistogram(filePath)
        distances[i]=calculDistance(h1, h)
        h1=h
        i += 1
    return distances
    
def createHtmlForImages(htmlFileName):
    fileNames = filesNamesInDirectory(imagesDirectoryPath)
    output_file = open(htmlFileName, 'w')
    output_file.write('<html>\n')
    output_file.write('<head>\n')
    output_file.write('</head>\n')
    output_file.write('<body>\n')
    output_file.write('<table border="2">\n')
    i=0
    for fileName in fileNames:
        if (i%5)==0:
            output_file.write('<tr>\n')
        i +=1
        output_file.write('<td  align="center"><img src="images/' + fileName + '"/><br/>'  + fileName +  '</td>\n')
        if (i%5)==0:
            output_file.write('</tr>\n')     
    output_file.write('</table>\n')
    output_file.write('</body>\n')
    output_file.write('</html>\n')
    output_file.close()

def saveNpArray(path, npArray):
    np.save(path, npArray)
    
    
def predict(distances, threshold):
    return 1*(distances>=threshold)

# def loss_function(x, y, threshold):
#     """ fonction de cout 0-1"""
#     return abs(y - predict(x, threshold))
    
def loss_function(Y_actual, Y_estimated):
    return np.dot(abs(np.subtract(Y_actual, Y_estimated)).T, np.ones(Y_actual.shape))

    
# ******************************************
#                 Main program
# ******************************************

# imagesDirectoryPath = "C:/data/SignalProcessing/images"
imagesDirectoryPath = "d:/Alireza/data/images"

# indexes of images correspoindig to a frame change (the first image is 1 and the last 150)
frameChangeIndexes = np.asarray([12, 13, 17, 19, 24, 29, 32, 35, 37, 46, 63, 68, 69, 72, 74, 80, 84, 86, 97, 105, 114, 117, 122, 137, 141, 147, 157, 160, 165, 175, 177, 180, 183, 195, 198, 201, 205, 213, 218, 226, 229, 236, 237, 243, 251, 257, 263, 268, 269, 272, 274, 279, 284, 288, 296, 298])

# converting to a zero starting list (from 0 to 149)
frameChangeIndexes = frameChangeIndexes -1

# Creating a learning data array with 1 for indexes corresponding to a frame change
trainingData = np.zeros(300)
trainingData[frameChangeIndexes]=1

# Getting distances for all images (1124 in total)
# The following 2 lines are commented because distances have already been saved on the disk
# distances = calculAllDistances(imagesDirectoryPath)
# saveNpArray("distances-data", distances)

# getting distances from the saved file
distances = np.load("distances-data.npy")

# Grabbing the first 150 images corresponding to the training data
trainingFeatures = distances[range(300)]

print "Frame change distances statistics :"
print "Mean = %f" % (np.mean(trainingFeatures[trainingData==1]))
print "Std = %f" % (np.std(trainingFeatures[trainingData==1]))
print "Median = %f " % (np.median(trainingFeatures[trainingData==1]))

print "No frame change distances statistics :"
print "Mean = %f" % (np.mean(trainingFeatures[trainingData==0]))
print "Std = %f" % (np.std(trainingFeatures[trainingData==0]))
print "Median = %f " % (np.median(trainingFeatures[trainingData==0]))

plt.hist(trainingFeatures[trainingData==0])
plt.hist(trainingFeatures[trainingData==1])


trainingFeatures = trainingFeatures.reshape(300,1)
trainingData = trainingData.reshape(300,1)

model = LogisticRegression()
model = model.fit(trainingFeatures, trainingData)

model.score(trainingFeatures, trainingData)

Y2 = model.predict(trainingFeatures)
L = loss_function(trainingData, Y2)


d2 = np.load("C:/Users/Alireza/Google Drive/work/FormationDS/Cours/Mod04_SignalProcessing/TP/distances-data.npy")

def loss_function_2(threshold):
    estimated_output = 1*(trainingFeatures>=threshold)
    delta = abs(np.subtract(estimated_output, trainingData)).T
    return np.dot(delta, np.ones(trainingFeatures.shape))

x_min = optimize.brent(loss_function_2)

thresholds = np.arange(0,50000,100)
losses = loss_function_2(thresholds)
plt.plot(thresholds, losses)
