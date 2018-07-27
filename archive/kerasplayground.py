from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import cifar10
from keras import utils
import numpy as np
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(type(X_train))
print(type(y_train))
print(X_train[0].shape)
print(X_train.shape)
import pydot_ng as pydot
from keras.utils import plot_model
#exit()

model = Sequential()
# Generate dummy data
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
# Convert labels to categorical one-hot encoding
one_hot_labels = utils.to_categorical(labels, num_classes=10)

model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
plot_model(model, show_shapes=True, to_file='model.png')
model.save('playground_model.h5')
