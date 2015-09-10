import numpy as np
import os
import cv2

# Fonction de parcours de repertoire

def filesNamesInDirectory(directory):
    for root, dirs, files in os.walk(directory):
        fileNames = [];
        for name in files:
            if name.endswith((".jpg")) :
                fileNames.append(name)
        return fileNames
        



# Fonctions de traitement d'image

def calculHistogram(imageName):
    im=cv2.imread(imageName)
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv],[0],None,[180],[0,180])

def calculDistance( sign1, sign2, flag=1 ):
    #verification des dimensions
    L1 = len(sign1)
    L2 = len(sign2)
    if (L1 == L2 ):
        #calcul de la distance ou de la similarite
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


# Fonction de creation de la page html    
def createHtmlForImages(outputFileName, imageNames, y_pred):
    output_file = open(outputFileName, 'w')
    output_file.write('<html>\n')
    output_file.write('<head>\n')
    output_file.write('</head>\n')
    output_file.write('<body>\n')
    output_file.write('<table border="2">\n')
    i=0
    for imageName in imageNames:
        if (i%5)==0:
            output_file.write('<tr>\n')
        output_file.write('<td  align="center"><img src="images/' + imageName + '"/>')
        if (y_pred[i]==1):
            output_file.write('<font color="#FF0000"><b>')
        output_file.write('<br/>')
        output_file.write(imageName)
        if (y_pred[i]==1):
            output_file.write('</b></font>')
        output_file.write('</td>\n')
        i +=1
        if (i%5)==0:
            output_file.write('</tr>\n')     
    output_file.write('</table>\n')
    output_file.write('</body>\n')
    output_file.write('</html>\n')
    output_file.close()


def loss_function(thresholds, X, Y):
    return [lossForOne(t, X, Y) for t in thresholds]

def lossForOne(threshold, X, Y):
    estimated_output = [int(x>threshold) for x in X]
    delta = abs(np.subtract(estimated_output, Y)).T
    return np.dot(delta, np.ones(X.shape))
