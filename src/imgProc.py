import cv2
from random import random
import numpy as np

# from functools import singledispatch

# 定数
REVERSAL_P = 0.1


class ImgProc:

    # imit内の変数すべてパブリックなので注意
    def __init__(self, path):
        self.img = cv2.imread(path)
        self.w, self.h, self.ch = self.img.shape

    def get_ising(self):
        img_ising = np.zeros((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                if self.img[i][j][0] == 255:
                    img_ising[i][j] = 1
                else:
                    img_ising[i][j] = -1
        return img_ising

    def get_pots(self, p_grad):
        img_pots = np.zeros((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                img_pots[i][j] = int((self.img[i][j][0] * p_grad / 256))
                # if self.img[i][j][0] == 255:
                #   img_pots[i][j] = 1
                # else:
                #    img_pots[i][j] = 0
        return img_pots

    def get_colpots(self, p_grad):
        img_colpots = np.zeros((self.h, self.w, 3))
        for i in range(self.h):
            for j in range(self.w):
                img_colpots[i][j][0] = int((self.img[i][j][0] * p_grad / 256))
                img_colpots[i][j][1] = int((self.img[i][j][1] * p_grad / 256))
                img_colpots[i][j][2] = int((self.img[i][j][2] * p_grad / 256))
        return img_colpots

    def get_colpots_hsv(self, p_grad):
        img_colpots_hsv = np.zeros((self.h, self.w, 3))
        for i in range(self.h):
            for j in range(self.w):
                img_colpots_hsv[i][j][0] = int((self.img[i][j][0] * 999 / 256))
                img_colpots_hsv[i][j][1] = int((self.img[i][j][1] * 999 / 256))
                img_colpots_hsv[i][j][2] = int((self.img[i][j][2] * 999 / 256))
        return img_colpots_hsv

    def get_img_size(self):
        return self.h, self.w

    def to_binary(self):
        ret, img_bin = cv2.threshold(self.img, 100, 255, cv2.THRESH_BINARY)
        self.img = img_bin
        return self.img

    def to_rand_noise(self):
        for i in range(self.h):
            for j in range(self.w):
                if REVERSAL_P > random():
                    rev_val = 255 - self.img[i][j][0]
                    self.img[i][j][0] = rev_val
                    self.img[i][j][1] = rev_val
                    self.img[i][j][2] = rev_val
        return self.img

    def to_rand_noise_color(self):
        for i in range(self.h):
            for j in range(self.w):
                if REVERSAL_P > random():
                    self.img[i][j][0] = int(random() * 256)
                    self.img[i][j][1] = int(random() * 256)
                    self.img[i][j][2] = int(random() * 256)
        return self.img

    def to_hsv(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        return self.img
