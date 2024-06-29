import cv2
import copy

from imgProc import ImgProc

path="img/Lighthouse.bmp"

img_org=ImgProc(path)
img_bin=ImgProc(path)

#2値化
img_bin.binary()

#img_bin.rand_noise().img

cv2.imshow("portrait",img_org.img)
cv2.waitKey(0)
cv2.imshow("portrait",img_bin.img)
cv2.waitKey(0)

img_org.rand_noise(0.1)
img_bin.rand_noise()

#cv2.imshow("portrait",img_org.img)
#cv2.waitKey(0)
cv2.imshow("portrait",img_bin.img)
cv2.waitKey(0)