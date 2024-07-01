import cv2
from random import random
#from functools import singledispatch

#定数
REVERSAL_P = 0.1

class ImgProc:

    #imit内の変数すべてパブリックなので注意
    def __init__(self,path):
        self.img=cv2.imread(path)
        self.w,self.h,self.ch=self.img.shape


    def binary(self):
        ret,img_bin=cv2.threshold(self.img,100,255,cv2.THRESH_BINARY)
        self.img=img_bin
        return self.img


    def rand_noise(self):
        for i in  range(self.h):
            for j in range(self.w):
                if REVERSAL_P > random():
                    rev_val = 255 - self.img[i][j][0]
                    self.img[i][j][0]=rev_val
                    self.img[i][j][1]=rev_val
                    self.img[i][j][2]=rev_val
        return self.img

