# Simple CNN model for CIFAR-10
import os
os.environ['KERAS_BACKEND']='theano'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.utils import np_utils, plot_model
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras import backend as K
#import pydot_ng as pydot
from output import DataOut
import numpy as np
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')


class ConvModel(object):
    def __init__(self,dataset,filters=16, filter_size=3, epochs=5,
                 lrate=0.01, batch_size=32, momentum=0.9, pooling=2,
                 dropout1=0.2, dropout2=0.5,activation_function1="relu",
                 activation_function2="softmax", layer_count=1, **kwargs):
        self.dataset = dataset
        self.dataout = DataOut()
        self.filters = filters
        self.filter_size = filter_size
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.activation_function1 = activation_function1
        self.activation_function2 = activation_function2
        self.pooling = pooling
        self.epochs = epochs
        self.lrate = lrate
        self.decay = self.lrate/self.epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.layer_count = layer_count
        self.sgd = SGD(lr=self.lrate, momentum=self.momentum, decay=self.decay, nesterov=False)

        # output
        self.out = DataOut()

        # Create the model
        self.model = Sequential()
        #print(self.dataset.shape)
        for layer in range(0,self.layer_count):
            self.model.add(Conv3D(self.filters, (self.filter_size, self.filter_size, self.filter_size), input_shape=self.dataset.shape, data_format="channels_last",
                 padding='same', activation=self.activation_function1, kernel_constraint=maxnorm(3)))
            self.model.add(Dropout(self.dropout1))
            #model.add(Conv2D(200, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
            self.model.add(MaxPooling3D(pool_size=(self.pooling, self.pooling, self.pooling)))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation=self.activation_function1, kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(self.dropout2))
        self.model.add(Dense(len(self.dataset.allshapes), activation=self.activation_function2))
        # Compile model
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=self.sgd, metrics=['accuracy'])
        self.stored_prediction_vals = []

    def fit(self, earlystop, logfilename=None, testFold=-1):
        # print dataset parameters before fitting the model
        print(str(self.dataset))
        # Handle callbacks
        callbacks = []
        if logfilename != None:
            csv_logger = CSVLogger(logfilename)
            callbacks.append(csv_logger)
        if earlystop != 0:
            callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=3, verbose=0, mode='auto'))
        # Fit the model
        # If a test fold was specified
        if testFold >= 0:
            xtrain,ytrain,xval,yval,xtest,ytest = self.dataset.getKFCVTrainTest(testFold)
            self.model.fit(xtrain, ytrain, validation_data=(xval,yval), epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks, shuffle=True)

        # else just use the designated training set
        else:
            xtrain,ytrain,xval,yval,xtest,ytest = self.dataset.getNonKFCVTrainTest()
            self.model.fit(xtrain, ytrain, validation_data=(xval,yval), epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks, shuffle=True)

    def predict(self, testFold=-1):
        # Final evaluation of the model
        #scores = self.model.evaluate(self.dataset.xtest, self.dataset.ytest, verbose=1)
        #print("Accuracy: %.2f%%" % (scores[1] * 100))
        if testFold >= 0:
            xtrain,ytrain,xval,yval,xtest,ytest = self.dataset.getKFCVTrainTest(testFold)
            #print(ytest)
            self.stored_prediction_vals = np.concatenate((self.model.predict(xtest, 30, 1), ytest), axis=1)
        else:
            xtrain,ytrain,xval,yval,xtest,ytest = self.dataset.getNonKFCVTrainTest()
            #print(ytest)
            self.stored_prediction_vals = np.concatenate((self.model.predict(xtest, 30, 1), ytest), axis=1)

        self.buildConfusionMatrix (self.stored_prediction_vals)

    def buildConfusionMatrix (self, prediction_vals) :
        nclasses = int (prediction_vals.shape[1] / 2);

        matrix = [[0 for i in range (nclasses)] for j in range (nclasses)]
        for result in prediction_vals :
            pred, actl = np.argmax (result[: nclasses]), np.argmax (result[nclasses:])
            matrix[pred][actl] += 1

        self.stored_prediction_confusion_matrix = np.array ([["{0}_prediction".format(shape)] + matrix[id] for (id, shape) in enumerate (self.dataset.allshapes)])

    def getPredictionAccuracies(self):
        val = -1.0
        numcorrect = -1
        total = -1
        if len(self.stored_prediction_vals) != 0:
            vecLen = len(self.stored_prediction_vals[0])
            dataLen = int(vecLen/2)
            confidences = []
            actual = []
            for i in self.stored_prediction_vals:
                c = []
                a = []
                for j in range(0, int(dataLen)):
                    c.append(i[j])
                    a.append(i[j+int(dataLen)])
                confidences.append(c)
                actual.append(a)

            numCorrect = 0
            for i in range(0,len(confidences)):
                if confidences[i].index(max(confidences[i])) == actual[i].index(max(actual[i])):
                    numCorrect += 1

        val = (numCorrect*1.0)/len(self.stored_prediction_vals)
        return numCorrect, len(self.stored_prediction_vals), val

    # def to_png(self, filename):
    #     plot_model(self.model, show_shapes=True, to_file=filename)

    def to_h5(self, filename):
        self.model.save(filename)

    def to_prediction_csv(self, filename):
        self.dataout.WriteToCSV(self.stored_prediction_vals, filename, ["{0}_prediction".format(shape) for shape in self.dataset.allshapes]+["{0}_true".format(shape) for shape in self.dataset.allshapes])
        self.dataout.WriteToCSV(self.stored_prediction_confusion_matrix, filename.split ('.')[0] + "_confusion_matrix.csv", [""] + ["{0}_true".format (shape) for shape in self.dataset.allshapes], '%s')

    def visualize(self, testFold=-1, root_name="default"):
        input_img = self.model.input
        fig = plt.figure(figsize=(5,5))

        num_filters = 0
        layer = self.model.layers[0]
        layer_output = layer.output
        num_filters = min (16, layer.output_shape[4])

        if testFold >= 0:
            xtrain,ytrain,xval,yval,xtest,ytest = self.dataset.getKFCVTrainTest(testFold)
        else:
            xtrain,ytrain,xval,yval,xtest,ytest = self.dataset.getNonKFCVTrainTest()

        def getClass (x) :
            for i in range (len (x)) :
                if x[i] == 1 :
                    return i
            return len (x)

        ypred = self.model.predict_classes (xtest, 30, 1)

        keep = {}
        for id_val in range (len (xtest)) :
            cl = getClass (ytest[id_val])
            input_img_data = np.squeeze(xtest[id_val:id_val + 1])
            ax = fig.add_subplot(111, projection='3d')
            z,x,y = input_img_data.nonzero()
            plt.axis('scaled')
            ax.set_xlim([0,input_img_data.shape[0]])
            ax.set_ylim([0,input_img_data.shape[1]])
            ax.set_zlim([0,input_img_data.shape[2]])
            ax.scatter(x, y, z, zdir='z', cmap='plasma',c=-y)
            # input_img_data = xtest[id_val:id_val + 1]
            # for i in range (input_img_data[0].shape[0]) :
            #     for j in range (input_img_data[0].shape[1]) :
            #         for k in range (input_img_data[0].shape[2]) :
            #             if input_img_data[0][i][j][k][0] > 0 :
            #                 ax.scatter (i, j, k, c='r')

            if ypred[id_val] == cl :
                plt.savefig (root_name + '_visual_correct_' + str (id_val) + '.png')
            else :
                plt.savefig (root_name + '_visual_wrong_' + str (id_val) + '.png')

            if cl in keep :
                continue
            keep[cl] = 1

            plt.savefig (root_name + '_visual_input_' + str (cl) + '.png')
            plt.clf ()

            # for filter_index in range(num_filters):
            #     loss = K.mean(layer_output[:, :, :, filter_index])
            #     grads = K.gradients(loss, input_img)[0]
            #
            #     iterate = K.function([input_img], [loss, grads])
            #
            #     step = 1.
            #
            #     input_img_data = xtest[id_val:id_val + 1]
            #     # we run gradient ascent for 20 steps
            #     for i in range(20):
            #         loss_value, grads_value = iterate([input_img_data])
            #         input_img_data += grads_value * step
            #
            #     ax = fig.add_subplot(4, 4, filter_index + 1, projection='3d')
            #
            #     d = []
            #     for i in range (input_img_data[0].shape[0]) :
            #         for j in range (input_img_data[0].shape[1]) :
            #             for k in range (input_img_data[0].shape[2]) :
            #                 d.append ((i, j, k, input_img_data[0][i][j][k]))
            #     d.sort (key=lambda x: x[3])
            #     x, y, z, c = [], [], [], []
            #     for a in d :
            #         x.append (a[0])
            #         y.append (a[1])
            #         z.append (a[2])
            #         c.append (a[3])
            #     ax.scatter (x, y, z, cmap='plasma',c=y)
            #
            # plt.savefig (root_name+'_visual_' + str (cl) + '.png')
            # plt.clf ()
