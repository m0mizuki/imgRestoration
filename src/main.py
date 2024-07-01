import cv2

from imgProc import ImgProc
from imgRes import res_inaba, get_img_bin

path = "img/Lighthouse.bmp"
#path = "img/8_8.bmp"

#img_org = ImgProc(path)
#cv2.imshow("portrait", img_org.img)
#cv2.waitKey(0)
#img_org.to_rand_noise()
#cv2.imshow("portrait", img_org.img)
#cv2.waitKey(0)


img_bin = ImgProc(path)

# 2値化
img_bin.to_binary()

cv2.imshow("portrait", img_bin.img)
cv2.waitKey(0)

img_bin.to_rand_noise()

cv2.imshow("portrait", img_bin.img)
cv2.waitKey(0)

tmp_h, tmp_w = img_bin.get_img_size()
tmp_ising_bfr = img_bin.get_ising()
print(tmp_ising_bfr)
tmp_ising_aft = res_inaba(tmp_ising_bfr, tmp_h, tmp_w)
print(tmp_ising_aft)
tmp_img_bin = get_img_bin(tmp_ising_aft, tmp_h, tmp_w)
print(tmp_img_bin)

print("修復完了")
cv2.waitKey(0)
cv2.imshow("portrait", tmp_img_bin)
cv2.waitKey(0)
