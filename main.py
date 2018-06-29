#! /usr/bin/python
# -*- coding: utf-8 -*-


from dataHandler import DataHandler
from network import Network
import layer
import activationFunctions
import mnist_loader

if __name__ == '__main__':
    EPOCHS = 1000
    ETA = 1.0
    PATH = "../mnist/"
    CKPT_DIR = "./ckpt/"
    MBS = 10 #mini_batch_size
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # print "treino", training_data

    dataHandler = DataHandler(PATH)
    dataHandler.load_training()

    sizes = [784, 16,16,10]
    network = Network(sizes, EPOCHS/10.0, CKPT_DIR, activationFunctions.sigmoid)
    network.SGD(training_data, EPOCHS, MBS, ETA, test_data)
