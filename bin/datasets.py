import numpy as np
from shapes import Sphere, Torus, Cube, SquareTorus, Pyramid, Gridspace
from scipy.stats import ortho_group
from random import choice, shuffle
import matplotlib.pyplot as plt


class Dataset(object):
    def __init__(self, kfcv = 0, folds = 2, fold = 1, validation_split=0, training_size=50, testing_size=30,
                 sampling_rate=100, sampling_floats=0, grid_resolution=.5, grid_radius=15,
                 train_classes=["sphere","torus"], test_classes=["sphere","torus"],
                 translation=0,rotation=0,transformation=0,noise=0,scale_randomness=0,scale=1,
                 poly_transformation=0, **kwargs):
        self.grid = Gridspace(stepsize=grid_resolution, radius=grid_radius)
        self.allshapes = list(set(train_classes) | set(test_classes))
        shapecount = len(self.allshapes)
        shapematrix = np.identity(shapecount, int)
        shapesclass = {"sphere":Sphere,
                       "torus":Torus,
                       "pyramid":Pyramid,
                       "cube":Cube,
                       "squaretorus":SquareTorus}
        self.shapes = {}
        for i, shapeclass in enumerate(self.allshapes):
            self.shapes[shapeclass]=shapesclass[shapeclass](classarray=shapematrix[i,:])
        self.train_classes=train_classes
        self.test_classes=test_classes
        self.translation = translation
        self.rotation = rotation
        self.transformation = transformation
        self.poly_transformation = poly_transformation
        self.translation = translation
        self.scale = scale
        self.scale_randomness = scale_randomness
        self.noise = noise
        self.sampling_rate = sampling_rate
        self.sampling_floats = sampling_floats
        self.shape = self.grid.shape + [1]
        self.validation_split=validation_split

        # k-fold cross validation variables
        self.kfcv = kfcv
        self.fold = fold
        self.folds = folds
        self.foldData = []
        self.xdata = None
        self.ydata = None
        print("Generating Data")
        if not kfcv:
            # 0 is voxel that doesn't contain anything
            # 1 is a voxel that contains a point
            xtrain = []
            ytrain = []
            xtest = []
            ytest = []

            for i in range(0, training_size):
                shape = self.shapes[choice(self.train_classes)]
                shapedata = np.expand_dims(self.get_shape(shape),4)
                #print(shapedata.shape)
                xtrain.append(shapedata)
                ytrain.append(shape.classarray)

            for i in range(0, testing_size):
                shape = self.shapes[choice(self.test_classes)]
                shapedata = np.expand_dims(self.get_shape(shape),4)
                #print(shapedata.shape)
                xtest.append(shapedata)
                ytest.append(shape.classarray)

            self.setNonKFCVDataToShuffled(xtrain,ytrain,xtest,ytest)
            print("Generating Data - Complete!")

        else:
            numData = training_size + testing_size
            classes = list(set(train_classes) | set(test_classes))
            xdata = []
            ydata = []

            for i in range(0, numData):
                shape = self.shapes[choice(classes)]
                shapedata = np.expand_dims(self.get_shape(shape),4)
                xdata.append(shapedata)
                ydata.append(shape.classarray)

            # store all of the data
            self.xdata = xdata
            self.ydata = ydata

            # generate the folds
            for i in range(0, folds):
                self.foldData.append([])
            #alldata = list(zip(xdata,ydata))
            #shuffle(alldata)
            #self.xdata[:] = zip(*xdata)
            #self.ydata[:] = zip(*ydata)

            # STRATIFICATION
            XYsorted = self.getSortedDataByClass(self.xdata, self.ydata)

            currFold = 0
            for i in range(0, len(xdata)):
                self.foldData[currFold].append([XYsorted[i][0], XYsorted[i][1]])
                currFold = (currFold + 1)%folds
            print("Generating Data - Complete!")

    def getKFCVTrainTest(self, testFold, returnAsList=False):
        if self.kfcv == 0:
            print("Error: cannot access folds if k-fold cross validation is not enabled.")
            exit()

        test = self.foldData[testFold]  #2d array
        train = []
        for i in range(0, len(self.foldData)):
            if i != testFold:
                train += self.foldData[i]

        xtrain = []
        ytrain = []
        xtest = []
        ytest = []
        for i in train:
            xtrain.append(i[0])
            ytrain.append(i[1])

        for i in test:
            xtest.append(i[0])
            ytest.append(i[1])

        # stratify the validation set
        validationSize = int(len(xtrain)*self.validation_split)
        sortedTrain = self.getSortedDataByClass(xtrain,ytrain)
        xtrain_and_xval = [[],[]]
        ytrain_and_yval = [[],[]]
        stepsize = int(len(xtrain)/validationSize)
        indices = np.arange(0,len(xtrain),stepsize)
        for i in range(0, len(xtrain)):
            if i in indices:
                xtrain_and_xval[1].append(sortedTrain[i][0])
                ytrain_and_yval[1].append(sortedTrain[i][1])
            else:
                xtrain_and_xval[0].append(sortedTrain[i][0])
                ytrain_and_yval[0].append(sortedTrain[i][1])

        # shuffle training and testing sets
        traindata = list(zip(xtrain_and_xval[0],ytrain_and_yval[0]))
        shuffle(traindata)
        finalTrainX, finalTrainY = zip(*traindata)

        # returns xtrain, ytrain, xtest, ytest
        if returnAsList:
            #return xtrain, ytrain, xtest, ytest
            return finalTrainX, finalTrainY, xtrain_and_xval[1], ytrain_and_yval[1], xtest, ytest
        else:
            #return np.array(xtrain), np.array(ytrain), np.array(xtest), np.array(ytest)
            return np.array(finalTrainX), np.array(finalTrainY), np.array(xtrain_and_xval[1]), np.array(ytrain_and_yval[1]), np.array(xtest), np.array(ytest)

    def getNonKFCVTrainTest(self):
        if self.kfcv == 1:
            print("Error: cannot use the method setNonKFCDataToShuffled if k-fold cross validation is enabled")

        return self.xtrain, self.ytrain, self.xtest, self.ytest

    def setNonKFCVDataToShuffled(self, xtrain, ytrain, xtest, ytest):
        if self.kfcv == 1:
            print("Error: cannot use the method setNonKFCDataToShuffled if k-fold cross validation is enabled")

        ctrain = list(zip(xtrain, ytrain))
        shuffle(ctrain)
        xtrain[:], ytrain[:] = zip(*ctrain)

        ctest = list(zip(xtest, ytest))
        shuffle(ctest)
        xtest[:], ytest[:] = zip(*ctest)

        self.xtrain = np.array(xtrain)
        self.ytrain = np.array(ytrain)
        self.xtest = np.array(xtest)
        self.ytest = np.array(ytest)

        #print(self.xtrain.shape)
        #print(self.ytrain.shape)

    def get_shape(self, shape, maxtries = 10000):
        shape_valid = False
        tries = 0
        while shape_valid == False and tries <= maxtries:
            try:
                shape.set_transforms(rotation=self.rotation,
                                     scale=self.scale,
                                     scale_randomness=self.scale_randomness,
                                     noise=self.noise,
                                     translation = self.translation,
                                     transformation = self.transformation,
                                     polyscale=self.poly_transformation)

                ###Non-linear transformation
                ### For next level or randomness, can random walk on different conjugate subgroups
                ### changing the parameters in shape class as we go
                shape.sample(self.sampling_rate)
                shapedata = shape.as_grid(self.grid, sampling_floats=self.sampling_floats, enforce = True)
                return shapedata
                #plt.clf()
            except:
                if tries == maxtries:
                    raise IOError("Could not resolve a shape in the grid within {0} tries".format(maxtries))
                else:
                    tries+=1
        return None

    def __str__(self):
        string = "DATASET PARAMETERS:\n"
        string += "k-Crossfold Validation?: " + str(self.kfcv) + "\n"
        if self.kfcv == 1:
            string += " --- Num folds: " + str(self.folds) + "\n"
            string += " --- Curr test fold (starting at 0): " + str(self.fold) + "\n"

        return string

    def kfcvPrintDataIDs(self):
        if self.kfcv != 1:
            print("Error: non-fatal: cannot call method kfcvPrinDataIDs() if not using k-fold cross validation")
            return
        print("ALl DATA: ")
        for d in range(0, len(self.xdata)):
            print(" --- " + str(id(self.xdata[d])) + " " + str(self.ydata[d]))

    def kfcvPrintFoldInfo(self):
        if self.kfcv != 1:
            print("Error: non-fatal: cannot call method kfcvPrinDataIDs() if not using k-fold cross validation")
            return
        print("FOLD DATA: ")
        for i in range(0, self.folds):
            print(" --- TEST FOLD = " + str(i))
            xtrain,ytrain,xval,yval,xtest,ytest = self.getKFCVTrainTest(i, True)
            print(" --- --- train:")
            for j in range(0, len(xtrain)):
                print(" --- --- " + str(id(xtrain[j])) + " " + str(ytrain[j]))
            print(" --- --- validation:")
            for j in range(0, len(xval)):
                print(" --- --- " + str(id(xval[j])) + " " + str(yval[j]))
            print(" --- --- test:")
            for j in range(0, len(xtest)):
                print(" --- --- " + str(id(xtest[j])) + " " + str(ytest[j]))

    def getSortedDataByClass(self, xdata, ydata):
        idxlist = []
        for y in ydata:
            ylist = y.tolist()
            idxlist.append(ylist.index(1))
        alldata = list(zip(xdata,ydata,idxlist))
        alldata.sort(key=lambda x:x[2])
        return alldata
