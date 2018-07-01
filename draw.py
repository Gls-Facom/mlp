#!/usr/bin/python

import cv2
import numpy as np

class Digits:
    def __init__(self,img):
        self.drawing = False # true if mouse is pressed
        self.ix,iy = -1,-1
        self.img = img

    def draw_circle(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix,self.iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.circle(self.img,(x,y),5,(0,0,0),-1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.circle(self.img,(x,y),5,(0,0,0),-1)

def get_image():
    img = np.zeros((128,128,1), np.uint8)
    img.fill(255)
    obj = Digits(img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',obj.draw_circle)
    imgp = cv2.resize(obj.img, (28,28))
    imgp.reshape(784,1)
    return imgp, obj        
