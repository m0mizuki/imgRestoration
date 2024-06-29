import cv2
import copy

from imgProc import ImgProc


ip=ImgProc("img/Lighthouse.bmp")

#2値化
img_bin=ip.binary()

#img_bin_cpy=copy.copy(img_bin)

img_bin_cpy=ip.rand_noise(1)


cv2.imshow("portrait",ip.img)
cv2.waitKey(0)
cv2.imshow("portrait",img_bin)
cv2.waitKey(0)
cv2.imshow("portrait",img_bin_cpy)
cv2.waitKey(0)