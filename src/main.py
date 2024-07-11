# 田中氏の論文「統計力学的手法をもとにした画像修復」参照

import cv2

from imgProc import ImgProc
from imgRes import res_tanaka, get_img_bin, get_img_grad

# 定数
POTS_GRAD = 2  # n値画像


path = "img/Lighthouse.bmp"
# path = "img/8_8.bmp"

# img_org = ImgProc(path)
# cv2.imshow("portrait", img_org.img)
# cv2.waitKey(0)
# img_org.to_rand_noise()
# cv2.imshow("portrait", img_org.img)
# cv2.waitKey(0)


img_bin = ImgProc(path)
img_org = ImgProc(path)

# 2値化
img_bin.to_binary()

cv2.imshow("portrait", img_bin.img)
cv2.waitKey(0)

img_bin.to_rand_noise()

cv2.imshow("portrait", img_bin.img)
cv2.waitKey(0)

tmp_h, tmp_w = img_bin.get_img_size()
# tmp_ising_bfr = img_bin.get_pots()
# tmp_ising_aft = res_tanaka(tmp_ising_bfr, tmp_h, tmp_w)
# tmp_img_bin = get_img_bin(tmp_ising_aft, tmp_h, tmp_w)
# print(tmp_img_bin)

cv2.imshow("portrait", img_org.img)
tmp_pots = img_org.get_pots(POTS_GRAD)
tmp_img_pots = get_img_grad(img_org.img, tmp_pots, POTS_GRAD, tmp_h, tmp_w)
print(tmp_img_pots)

print("修復完了")
cv2.waitKey(0)
# cv2.imshow("portrait", tmp_img_bin)
cv2.imshow("portrait", tmp_img_pots)
cv2.waitKey(0)
