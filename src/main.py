import cv2
import copy

from imgProc import ImgProc

path="img/Lighthouse.bmp"

img_org=ImgProc(path)
img_bin=ImgProc(path)

#2値化
img_bin.to_binary()

cv2.imshow("portrait",img_org.img)
cv2.waitKey(0)
cv2.imshow("portrait",img_bin.img)
cv2.waitKey(0)

#a=img_bin.get_ising()
#print(a)

img_org.to_rand_noise()
img_bin.to_rand_noise()

cv2.imshow("portrait",img_org.img)
cv2.waitKey(0)
cv2.imshow("portrait",img_bin.img)
cv2.waitKey(0)