import numpy as np
import matplotlib.pyplot as plt
import os


def filesNamesInDirectory(directory):
    for root, dirs, files in os.walk(directory):
        fileNames = [];
        for name in files:
            if name.endswith((".jpg")) :
                fileNames.append(name)
        return fileNames


# imagesDirectoryPath = "C:/data/SignalProcessing/images"
imagesDirectoryPath = "d:/Alireza/data/images"

fileNames = filesNamesInDirectory(imagesDirectoryPath)




# indexes of images correspoindig to a frame change (the first image is 1 and the last 150)
frameChangeIndexes = np.asarray([12, 13, 17, 19, 24, 29, 32, 35, 37, 46, 63, 68, 69, 72, 74, 80, 84, 86, 97, 105, 114, 117, 122, 137, 141, 147, 157, 160, 165, 175, 177, 180, 183, 195, 198, 201, 205, 213, 218, 226, 229, 236, 237, 243, 251, 257, 263, 268, 269, 272, 274, 279, 284, 288, 296, 298])

# converting to a zero starting list (from 0 to 149)
frameChangeIndexes = frameChangeIndexes -1

# Creating a learning data array with 1 for indexes corresponding to a frame change
Y_train = np.zeros(300)
Y_train[frameChangeIndexes]=1


X = np.load("C:/Users/Alireza/Google Drive/work/FormationDS/Cours/Mod04_SignalProcessing/TP/distances-data.npy")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

X_std_train = X_std[:300]


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

X_std_train = X_std_train.reshape(X_std_train.shape[0],1)
logreg.fit(X_std_train, Y_train)
Y_train_pred = logreg.predict(X_std_train)

X_std = X_std.reshape(X_std.shape[0],1)
Y_pred = logreg.predict(X_std)

I = range(len(Y_pred))
I[Y_pred==1.]







