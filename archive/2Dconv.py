# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from shapes import Sphere
from shapes import Torus
import numpy as np

sphere = Sphere()
torus = Torus()

# 0 is voxel that doesn't contain anything
# 1 is a voxel that contains a point
X_train = []
y_train = []
X_test = []
y_test = []

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

#print np.shape(xtrain)

# Create the model
model = Sequential()
model.add(Conv2D(200, (3, 3), input_shape=(32, 32,1), data_format="channels_last", padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(200, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(xtest, ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
	