##### scikit - learn . tutorials. #2
# source  https://scikit-learn.org/stable/auto_examples/plot_multioutput_face_completion.html#sphx-glr-auto-examples-plot-multioutput-face-completion-py

import numpy as np              
import matplotlib.pyplot as plt

#importing sklearn libraries
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


######################################################
# Load the faces datasets 

data = fetch_olivetti_faces('.')   #Use data from currently using folder.
targets = data.target

print('it was ',type(data)) #here data is still sklearn units

data = data.images.reshape((len(data.images),-1))

print('now it is ',type(data))  #now we can see easily it is numpy array

######################################################
# Separeting data for training and testing
train = data[targets < 30]
print((len(train)/len(data)),'% of data for training')
test = data[targets >= 30]
print((len(test)/len(data)),'% of data for test')

######################################################
# Test on a subset of people

nFaces = 5
rng = check_random_state(4)
faceIds = rng.randint(test.shape[0], size=(nFaces, ))
test = test[faceIds, :]

n_pixels = data.shape[1]
#upper half of the faces 
XTrain = train[:, :(n_pixels + 1) // 2]
#lower half of the faces 
yTrain = train[:, n_pixels // 2: ]

#Preparing test data 
xTest = test[:, :(n_pixels+1) // 2]
yTest = test[:, n_pixels // 2: ]

####################################################
# Fit estimators algorithms.

estimators = {
        "Extra Trees":ExtraTreesRegressor(n_estimators=10,max_features=32,
                                         random_state=0),
        "K-nn": KNeighborsRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge": RidgeCV(),
}


yTestPredict = dict()

for name, estimator in estimators.items():      #loop on algorithm list
    estimator.fit(XTrain,yTrain)
    yTestPredict[name] = estimator.predict(xTest)
    
####################################################
#    Plot the completed faces 

imageShape=(64, 64)

nCols = 1 + len(estimators)
plt.figure(figsize=(2. * nCols, 2.26 * nFaces))
plt.suptitle("Face completion with multi output estimators", size=15)

for i in range(nFaces):
    trueFace = np.hstack((xTest[i], yTest[i]))
    if i:
        sub = plt.subplot(nFaces, nCols, i * nCols +1)
    else:
        sub = plt.subplot(nFaces, nCols, i * nCols +1,
                          title = "True Faces")
    sub.axis("off")
    sub.imshow(trueFace.reshape(imageShape),
               cmap = plt.cm.gray,
               interpolation = 'nearest')
    for j, est in enumerate(sorted(estimators)):
        completed_face = np.hstack((xTest[i], yTestPredict[est][i]))
        
        if i:
            sub = plt.subplot(nFaces, nCols, i * nCols + 2 + j)
        else:
            sub = plt.subplot(nFaces, nCols, i * nCols + 2 + j,
                             title = est)
        sub.axis('off')
        sub.imshow(completed_face.reshape(imageShape),
                  cmap=plt.cm.gray,
                  interpolation='nearest')
plt.show()
