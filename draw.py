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
            # print "TOIS"
            self.drawing = False
            cv2.circle(self.img,(x,y),5,(0,0,0),-1)
