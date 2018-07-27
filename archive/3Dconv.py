# Simple CNN model for CIFAR-10
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from shapes import Sphere
from shapes import Torus
from shapes import Gridspace
import numpy as np
from scipy.stats import ortho_group
import pydot_ng as pydot
from keras.utils import plot_model

# NET PARAMETERS
stride=(2,2,2)
pad='same'
filters=16
lrate=.1
epochs=5

#sphere = Sphere()
#torus = Torus()
grid = Gridspace(stepsize=.5, radius=15)
mysphere = Sphere(dim=3, radius=5)
mytorus = Torus(dim=3, major_radius=4, minor_radius=2)

# RANDOMLY GENERATE DATA
# 0 is voxel that doesn't contain anything
# 1 is a voxel that contains a point
X_train = []
y_train = []
X_test = []
y_test = []

for i in range(0, 5):
    mysphere.transformationmatrix = ortho_group.rvs(dim=3) + np.random.normal(0,.05,(3,3))
    mytorus.transformationmatrix = ortho_group.rvs(dim=3) + np.random.normal(0,.05,(3,3))
    mysphere.translationvector = np.random.normal(0,3,3)
    mytorus.translationvector = np.random.normal(0,3,3)
    myspherepoints = mysphere.sample(20)
    mytoruspoints = mytorus.sample(20)
    myspheregrid = mysphere.as_grid(grid)
    desiredsphereshape = myspheregrid.shape + (1,)
    myspheregrid = myspheregrid.reshape(*desiredsphereshape)
    mytorusgrid = mytorus.as_grid(grid)
    desiredtorusshape = mytorusgrid.shape + (1,)
    mytorusgrid = mytorusgrid.reshape(*desiredtorusshape)
    class1 = np.array([0, 1])
    class2 = np.array([1, 0])

    if i < 4:
        X_train.append(myspheregrid)
        y_train.append(class1)
        X_train.append(mytorusgrid)
        y_train.append(class2)

    if i >= 4:
        X_test.append(myspheregrid)
        y_test.append(class1)
        X_test.append(mytorusgrid)
        y_test.append(class2)

xtrain = np.array(X_train)
ytrain = np.array(y_train)
xtest = np.array(X_test)
ytest = np.array(y_test)

'''
for i in range(0,50):
	image1 = sphere.sampleToImage(1000, "sphere.png")
	class1 = np.array([0,1])
	image2 = torus.sampleToImage(1000, "torus.png")
	class2 = np.array([1,0])

	if i < 45:
		X_train.append(image1)
		y_train.append(class1)
		X_train.append(image2)
		y_train.append(class2)
	else:
		X_test.append(image1)
		y_test.append(class1)
		X_test.append(image2)
		y_test.append(class2)

xtrain = np.array(X_train)
ytrain = np.array(y_train)
xtest = np.array(X_test)
ytest = np.array(y_test)
'''

# Create the model
model = Sequential()
model.add(Conv3D(filters, (3, 3, 3), strides=stride, input_shape=myspheregrid.shape, data_format="channels_last",
                 padding=pad, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
#model.add(Conv2D(200, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# Compile model
decay = lrate / epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(xtrain, ytrain, validation_data=(
    xtest, ytest), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(xtest, ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
plot_model(model, show_shapes=True, to_file='model.png')
#model.save('conv3D_model.h5')
