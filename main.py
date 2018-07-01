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
        print "I'VE BEEN THROUGH THE DESERT ON A HORSE WITH NO ", learning_path
        network.load_learning(learning_path)
        img = np.zeros((56,56,1), np.uint8)
        img.fill(255)
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
                # print imgp.shape
                imgp = imgp.reshape(784,1)/255.0
                print network.predict(imgp)

                # break

    else:
        print "ENTÃO ME AJUDE A SEGURAR"
        network.SGD(training_data, int(EPOCHS), int(MBS), ETA, test_data)
