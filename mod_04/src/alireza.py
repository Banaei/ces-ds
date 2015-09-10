import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# "/home/userfc/Desktop/Alireza/CES-tp"
imagesDirectoryPath = "C:/data/SignalProcessing/images"

# Creation d'une image aléatoire
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)
# Convert the array to make a 400x300 grayscale image.
grayImage = flatNumpyArray.reshape(300, 400)
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite("bgrImage.jpg", bgrImage)
plt.figure
plt.subplot(2,1,1)
plt.imshow(bgrImage)
plt.title('image en couleur')
plt.subplot(2,1,2)
plt.imshow(grayImage, cmap=pyplot.cm.binary) #attention il y a quelque chose à ajouter !
plt.title('image en niveau de gris')


# Rotation de l'image
originalImage = cv2.imread('00000002.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

#rotation de l'image de 45 degrés
rows,cols = originalImage.shape

M = cv2.getRotationMatrix2D((rows/2,cols/2),45,1)
rotatedImage = cv2.warpAffine(originalImage,M,(cols,rows))

plt.figure(0)
plt.subplot(2,2,1)
plt.imshow(originalImage)
plt.title('image originale')
plt.subplot(2,1,2)
plt.imshow(rotatedImage)
plt.title('image tournee')

#qu'observez-vous ?

m = np.mean(originalImage)
mr = np.mean(rotatedImage)
v= np.var(originalImage)
vr = np.var(rotatedImage)

plt.figure(1)
x=[m, mr]
y=[v, vr]
plt.plot(m, v, '.')
plt.annotate('orig', [m,v])
plt.plot(mr, vr, '.')
plt.annotate('tournee', [mr,vr])
plt.title('exploration de l''espace des attributs')


# Calcul des moyennes et variance sur toutes les images
def getMeanAndVar(fileName):
    img = cv2.imread(fileName)
    m = np.mean(img)
    v = np.var(img)
    return m, v

for root, dirs, files in os.walk():
    for name in files:
        if name.startswith(("000000")) and name.endswith((".jpg")) :
            print name
            print getMeanAndVar(name)
            
            
# Séparation des caneaux
image = cv2.imread("00000005.jpg")
rows, col, nbChannels = image.shape
b, g, r = cv2.split(image) #cv2 et plt ne travaillent pas dans le même ordre des couleurs !!
image = cv2.merge([r, g, b])


plt.figure()
plt.subplot(2,2,1)
plt.imshow(image)
plt.title('image en couleur')
plt.subplot(2,2,2)
plt.imshow(b, cmap='gray')
plt.title('canal bleu')
plt.subplot(2,2,3)
plt.imshow(g, cmap='gray')
plt.title('canal vert')
plt.subplot(2,2,4)
plt.imshow(r, cmap='gray')
plt.title('canal rouge')

# Comparaison des histogrammes

def changeRgbForPlot(img):
    rows, col, nbChannels = image.shape
    r, g, b = cv2.split(image) 
    return cv2.merge([r, g, b])
    
    
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
        

refImageName = "00000005.jpg"
ditances = [[]]
fileNames = [[]]
h1 = calculHistogram(refImageName)
for root, dirs, files in os.walk(imagesDirectoryPath):
    for name in files:
        if name.startswith(("000000")) and name.endswith((".jpg")) :
            h = calculHistogram(name)
            ditances.append(calculDistance(h1, h))
            fileNames.append(name)

ind = np.argsort(ditances)     
       
plt.figure();

# plt.subplot(2,3,0) #pareil que plt.subplot(2,3,6)
plt.imshow(changeRgbForPlot(cv2.imread(refImageName)))
plt.title('Image de référence')
plt.show()

for i in range(0,len(fileNames)-1):
    n = ind[i]
#     plt.subplot(12, 2,i)
    filename = fileNames[n]
    plt.figure();
    plt.imshow(changeRgbForPlot(cv2.imread(filename)))
    plt.title('{}:{}'.format(i,filename))
    plt.show()
    

