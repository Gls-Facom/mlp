#! /usr/bin/python
# -*- coding: utf-8 -*-

from dataHandler import DataHandler
from network import Network
from config import load_config
import layer
import activationFunctions
import mnist_loader
import sys

if __name__ == '__main__':
    sizes, EPOCHS, MBS, ETA, activation_function = load_config()

    # PATH = "../mnist/"
    CKPT_DIR = "./ckpt/"
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # dataHandler = DataHandler(PATH)
    # dataHandler.load_training()

    network = Network(sizes, EPOCHS/10.0, CKPT_DIR, activation_function)
    network.SGD(training_data, int(EPOCHS), int(MBS), ETA, test_data)
