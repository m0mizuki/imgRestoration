# 田中氏の論文「統計力学的手法をもとにした画像修復」参照

import cv2

from imgProc import ImgProc
from imgRes import (
    res_tanaka,
    res_metropolis,
    res_metropolis_col,
    get_img_bin,
    get_img_grad,
    get_img_colgrad,
    res_heikinnka,
    res_gaussian,
    res_median,
)

# 定数
POTS_GRAD = 4  # n値画像,TA_POTSに等しい

path = "img/parrots.jpg"
# path = "img/Lighthouse.bmp"
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

# cv2.imshow("portrait", img_bin.img)
cv2.imshow("portrait", img_org.img)
cv2.waitKey(0)

# img_bin.to_rand_noise()
img_org.to_rand_noise_color()

# cv2.imshow("portrait", img_bin.img)
cv2.imshow("portrait", img_org.img)
print(img_org.img)
cv2.waitKey(0)

tmp_h, tmp_w = img_org.get_img_size()

# tmp_h, tmp_w = img_bin.get_img_size()
# tmp_ising_bfr = img_bin.get_pots()
# tmp_ising_aft = res_tanaka(tmp_ising_bfr, tmp_h, tmp_w)
# tmp_img_bin = get_img_bin(tmp_ising_aft, tmp_h, tmp_w)
# print(tmp_img_bin)

# tmp_pots_bfr = img_org.get_pots(POTS_GRAD)
# cv2.imshow("portrait", get_img_grad(img_org.img, tmp_pots_bfr, POTS_GRAD, tmp_h, tmp_w))
# cv2.waitKey(0)
## tmp_pots_aft = res_tanaka(tmp_pots_bfr, tmp_h, tmp_w)
# tmp_pots_aft = res_metropolis(tmp_pots_bfr, tmp_h, tmp_w)
#tmp_img_pots = get_img_grad(img_org.img, tmp_pots_aft, POTS_GRAD, tmp_h, tmp_w)

tmp_pots_bfr = img_org.get_colpots(POTS_GRAD)
cv2.imshow("portrait", get_img_colgrad(img_org.img, tmp_pots_bfr, POTS_GRAD, tmp_h, tmp_w))
cv2.waitKey(0)
tmp_pots_aft = res_metropolis_col(tmp_pots_bfr, tmp_h, tmp_w)
tmp_img_pots = get_img_colgrad(img_org.img, tmp_pots_aft, POTS_GRAD, tmp_h, tmp_w)
# print(tmp_img_pots)


print("修復完了")
cv2.waitKey(0)
# cv2.imshow("portrait", tmp_img_bin)
cv2.imshow("portrait", tmp_img_pots)
cv2.waitKey(0)

# tmp_img_grad = get_img_grad(img_org.img, tmp_pots_bfr, POTS_GRAD, tmp_h, tmp_w)
# print("平均化フィルタ")
# tmp_img_heikinka = res_heikinnka(tmp_img_grad, tmp_h, tmp_w)
# cv2.imshow("portrait", tmp_img_heikinka)
# cv2.waitKey(0)
# print("ガウシアンフィルタ")
# tmp_img_gaussian = res_gaussian(tmp_img_grad, tmp_h, tmp_w)
# cv2.imshow("portrait", tmp_img_gaussian)
# cv2.waitKey(0)
# print("メディアンフィルタ")
# tmp_img_median = res_median(tmp_img_grad, tmp_h, tmp_w)
# cv2.imshow("portrait", tmp_img_median)
# cv2.waitKey(0)
