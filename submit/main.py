from shapes import Sphere, Torus, Pyramid, Cube, SquareTorus, Gridspace
from datasets import Dataset
from models import ConvModel
from docopt import docopt
from graph import Graph
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ortho_group
import json
import random
import gc

__version__ = "0.1"
__doc__ = """ShapeDeepLearning v{0}

Usage:
     main.py [options] (-c <file> | --config <file>)

Options:
     -h --help                            Show this screen. [default: False]
     -v --version                         Show the current version. [default: False]
     -c <file> --config <file>            Specify the config file.
""".format(__version__)

def runNet(neuralnet, config, testFold=-1):
    neuralnet.fit(config["earlystop"], "fold"+str(testFold)+"_{root_name}_epoch_logs.csv".format(**config), testFold)
    neuralnet.predict(testFold)
    numCorrect, total, accuracy = neuralnet.getPredictionAccuracies()
    print("Total predicted: " + str(total) + "\nNum Correct: " + str(numCorrect) + "\nAccuracy: " + str(accuracy))
    #neuralnet.to_png("fold"+str(testFold) + "_{root_name}_model.png".format(**config))
    if config["save_model"]:
        neuralnet.to_h5("fold"+str(testFold) + "_{root_name}.h5".format(**config))
    neuralnet.to_prediction_csv("fold"+str(testFold)+"_{root_name}_prediction_results.csv".format(**config))
    if config["visualize"] == 1:
        neuralnet.visualize(testFold=testFold, root_name=config["root_name"])

def checkConfigParams(config):
    if config["kfcv"] == 1 and config["folds"] < 2:
        print("Error: number of folds must be greater than 2 if using k-fold CV")
        exit()
    if config["kfcv"] == 1 and config["fold"] >= config["folds"]:
        print("Error: fold must be less than the number of folds. Perhaps it wasn't zero-based?")
        exit()

def run(config):
    print(config)
    checkConfigParams(config)
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    dataset = Dataset(**config)
    # preconditions: the dataset has generated a pool of data, and the folds have been generated
    if config["kfcv"] == 1 and config["kfcv_serial"] == 1:
        for i in range(0, config["folds"]):
            neuralnet = ConvModel(dataset,**config)
            #validate that the folds work
            #dataset.kfcvPrintFoldInfo()
            #xit()
            gc.collect()
            runNet(neuralnet, config, i)

    # preconditions: the dataset has generated a pool of data, and the folds have been generated
    elif config["kfcv"] == 1:
        neuralnet = ConvModel(dataset,**config)
        runNet(neuralnet, config)

    # preconditions: the dataset has generated distinct training and testing set
    else:
        neuralnet = ConvModel(dataset,**config)
        runNet(neuralnet, config)


if __name__ == '__main__':
    """
    The main function called when main.py is run
    from the command line:

    > python main.py
    """
    arguments = docopt(__doc__, version='ShapeDeepLearning v{0}'.format(__version__))
    with open(arguments["--config"]) as data_file:
        config = json.load(data_file)
    run(config)
