#! /usr/bin/python

from dataHandler import DataHandler
from network import Network
import layer
import activationFunctions

if __name__ == '__main__':
    EPOCHS = 100
    ETA = 100
    PATH = "/home/vinicios/reps/mnist/"
    MBS = 100 #mini_batch_size
    def SGD(self, training_data, epochs, mini_batch_size, eta, val_data=None):

    dataHandler = DataHandler(PATH)
    dataHandler.load_training()

    sizes = [16,16,10]
    network = Network(sizes, dataHandler)
    network.SGD()

    main()
