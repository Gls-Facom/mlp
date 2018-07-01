#! /usr/bin/python
# -*- coding: utf-8 -*-

from network import Network
from config import load_config
import layer
import activationFunctions
import mnist_loader
import sys
from draw import Digits
import cv2
import numpy as np

if __name__ == '__main__':
    sizes, EPOCHS, MBS, ETA, activation_function, learning_path = load_config()

    CKPT_DIR = "./ckpt/"
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    network = Network(sizes, EPOCHS/10.0, CKPT_DIR, activation_function)

    if learning_path != None:
        network.load_learning(learning_path)
        img = np.zeros((128,128,1), np.uint8)
        img.fill(0)
        obj = Digits(img)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',obj.draw_circle)
        print "Draw the number and press p to predict\n"
        while(1):
            cv2.imshow('image',obj.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord('p'):
                imgp = cv2.resize(obj.img, (28,28))
                imgp = imgp.reshape(784,1)/255.0
                np.set_printoptions(threshold=np.inf)
                print network.predict(imgp)
                img.fill(0)
            elif k == ord('c'):
                img.fill(0)
                # break

    else:
        network.SGD(training_data, int(EPOCHS), int(MBS), ETA, test_data)
