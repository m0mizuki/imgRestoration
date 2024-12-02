import math
import copy
import numpy as np
from random import random

import cv2

# 定数
# J = 0.4
J = 0.50
T_MAX = 2
T_MIN = 1
T_DELTA = 0.1
Q = 0.1
K = math.log((1 - Q) / Q) / 2
# THRESHOLD = 0.000001
THRESHOLD = 1.0  # 許容誤差

# tanaka
TA_POTS_Q = 8  # ポッツモデルの状態の数
TA_J = 0.50
TA_R = 3  # 反復回数
TA_TH = 1.0  # 許容誤差
TA_C = 1.0  # 温度の係数

# metropolis
METR_J = 1.0
MERT_K = 2.0
METR_BETA = 8.0
METR_CNT = 65536


# 引数は数値のみ取得されている(参照渡しではない)
# s[][]:修復画像データ(ising)
# g[][]:受信画像データ(ising)
def res_inaba(g, h, w):
    s = copy.copy(g)
    t = T_MAX
    a = np.zeros((h, w))

    while t > T_MIN:
        print(t)
        for i in range(h):
            for j in range(w):
                a[i][j] = s[i][j]
        t = t - T_DELTA

        diff = 1.0
        while diff >= THRESHOLD:
            # print(diff)
            b = np.zeros((h, w))
            diff = 0.0
            for i in range(h):
                for j in range(w):
                    a_sum = 0
                    if i != 0:
                        a_sum += a[i - 1][j]
                    if i != h - 1:
                        a_sum += a[i + 1][j]
                    if j != 0:
                        a_sum += a[i][j - 1]
                    if j != w - 1:
                        a_sum += a[i][j + 1]
                    b[i][j] = math.tanh(K * g[i][j] / t + J * a_sum / t)

                    diff += abs(b[i][j] - a[i][j])

            for i in range(h):
                for j in range(w):
                    a[i][j] = b[i][j]

        print(a)

        for i in range(h):
            for j in range(w):
                if a[i][j] < 0:
                    s[i][j] = -1
                else:
                    s[i][j] = 1

    return s


# 本来sum_eの差で判断するが、平均場近似でsum_e=各サイトのeの和
# としているので、変更した画素とその4近傍のeの変化のみを比較すればよい
def res_metropolis(g, h, w):
    print("開始")
    s = copy.copy(g)
    u = np.zeros((h, w))
    e = np.zeros((h, w))

    a, b, c = 0, 0, 0
    aa, bb, cc = 0.0, 0.0, 0.0

    # e_sum = 0
    for i in range(h):
        for j in range(w):
            sum = 0
            if i != 0:
                sum += diff_rate(s[i][j], s[i - 1][j], TA_POTS_Q)
            if i != h - 1:
                sum += diff_rate(s[i][j], s[i + 1][j], TA_POTS_Q)
            if j != 0:
                sum += diff_rate(s[i][j], s[i][j - 1], TA_POTS_Q)
            if j != w - 1:
                sum += diff_rate(s[i][j], s[i][j + 1], TA_POTS_Q)

            e[i][j] = -METR_J * sum - MERT_K * diff_rate(s[i][j], g[i][j], TA_POTS_Q)
            # e_sum += e[i][j]

    for n in range(METR_CNT * TA_POTS_Q):
        # ランダムに選んだ画素をランダムな値に変更するパターン
        # alt_i, alt_j = int(random() * h), int(random() * w)
        # alt_val = int(random() * TA_POTS_Q)

        # 全走査するパターン
        m = int(n / TA_POTS_Q)
        alt_i, alt_j = m % 256, int(m / 256)
        alt_val = n % TA_POTS_Q
        # alt_val = (TA_POTS_Q - 1) - (n % TA_POTS_Q)
        # alt_val = int(random() * TA_POTS_Q)

        e_diff = 0

        sum = 0
        if alt_i != 0:
            sum += diff_rate(alt_val, s[alt_i - 1][alt_j], TA_POTS_Q)
        if alt_i != h - 1:
            sum += diff_rate(alt_val, s[alt_i + 1][alt_j], TA_POTS_Q)
        if alt_j != 0:
            sum += diff_rate(alt_val, s[alt_i][alt_j - 1], TA_POTS_Q)
        if alt_j != w - 1:
            sum += diff_rate(alt_val, s[alt_i][alt_j + 1], TA_POTS_Q)
        ep = -METR_J * sum - MERT_K * diff_rate(alt_val, g[alt_i][alt_j], TA_POTS_Q)
        e_diff += ep - e[alt_i][alt_j]

        if alt_i != 0:
            sum = 0
            if alt_i - 1 != 0:
                sum += diff_rate(s[alt_i - 1][alt_j], s[alt_i - 2][alt_j], TA_POTS_Q)
            if alt_i - 1 != h - 1:
                sum += diff_rate(s[alt_i - 1][alt_j], alt_val, TA_POTS_Q)
            if alt_j != 0:
                sum += diff_rate(
                    s[alt_i - 1][alt_j], s[alt_i - 1][alt_j - 1], TA_POTS_Q
                )
            if alt_j != w - 1:
                sum += diff_rate(
                    s[alt_i - 1][alt_j], s[alt_i - 1][alt_j + 1], TA_POTS_Q
                )
            ep = -METR_J * sum - MERT_K * diff_rate(
                s[alt_i - 1][alt_j], g[alt_i - 1][alt_j], TA_POTS_Q
            )
            e_diff += ep - e[alt_i - 1][alt_j]

        if alt_i != h - 1:
            sum = 0
            if alt_i + 1 != 0:
                sum += diff_rate(s[alt_i + 1][alt_j], alt_val, TA_POTS_Q)
            if alt_i + 1 != h - 1:
                sum += diff_rate(s[alt_i + 1][alt_j], s[alt_i + 2][alt_j], TA_POTS_Q)
            if alt_j != 0:
                sum += diff_rate(
                    s[alt_i + 1][alt_j], s[alt_i + 1][alt_j - 1], TA_POTS_Q
                )
            if alt_j != w - 1:
                sum += diff_rate(
                    s[alt_i + 1][alt_j], s[alt_i + 1][alt_j + 1], TA_POTS_Q
                )
            ep = -METR_J * sum - MERT_K * diff_rate(
                s[alt_i + 1][alt_j], g[alt_i + 1][alt_j], TA_POTS_Q
            )
            e_diff += ep - e[alt_i + 1][alt_j]

        if alt_j != 0:
            sum = 0
            if alt_i != 0:
                sum += diff_rate(
                    s[alt_i][alt_j - 1], s[alt_i - 1][alt_j - 1], TA_POTS_Q
                )
            if alt_i != h - 1:
                sum += diff_rate(
                    s[alt_i][alt_j - 1], s[alt_i + 1][alt_j - 1], TA_POTS_Q
                )
            if alt_j - 1 != 0:
                sum += diff_rate(s[alt_i][alt_j - 1], s[alt_i][alt_j - 2], TA_POTS_Q)
            if alt_j - 1 != w - 1:
                sum += diff_rate(s[alt_i][alt_j - 1], alt_val, TA_POTS_Q)
            ep = -METR_J * sum - MERT_K * diff_rate(
                s[alt_i][alt_j - 1], g[alt_i][alt_j - 1], TA_POTS_Q
            )
            e_diff += ep - e[alt_i][alt_j - 1]

        if alt_j != w - 1:
            sum = 0
            if alt_i != 0:
                sum += diff_rate(
                    s[alt_i][alt_j + 1], s[alt_i - 1][alt_j + 1], TA_POTS_Q
                )
            if alt_i != h - 1:
                sum += diff_rate(
                    s[alt_i][alt_j + 1], s[alt_i + 1][alt_j + 1], TA_POTS_Q
                )
            if alt_j + 1 != 0:
                sum += diff_rate(s[alt_i][alt_j + 1], alt_val, TA_POTS_Q)
            if alt_j + 1 != w - 1:
                sum += diff_rate(s[alt_i][alt_j + 1], s[alt_i][alt_j + 2], TA_POTS_Q)
            ep = -METR_J * sum - MERT_K * diff_rate(
                s[alt_i][alt_j + 1], g[alt_i][alt_j + 1], TA_POTS_Q
            )
            e_diff += ep - e[alt_i][alt_j + 1]
            # print(ep)
            # print(e[alt_i][alt_j + 1])

        # ほんとは前の処理全体にこのif文かける
        if alt_val != s[alt_i][alt_j]:
            if e_diff <= 0:
                u[alt_i][alt_j] += alt_val
                a += 1
                aa += alt_val
            else:
                rev_p = math.exp(-METR_BETA * e_diff)
                b += 1
                bb += alt_val
                if rev_p > random():
                    u[alt_i][alt_j] += alt_val
                    c += 1
                    cc += alt_val
                else:
                    u[alt_i][alt_j] += s[alt_i][alt_j]

        if n % (4096 * TA_POTS_Q) == 0:
            print(int(n / (4096 * TA_POTS_Q)), "/16")

    for i in range(h):
        for j in range(w):
            u[i][j] = int(u[i][j] / (TA_POTS_Q - 1))

    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("aa/a:", aa / a)
    print("bb/b:", bb / b)
    # print("cc/c:", cc / c)

    return u


# s[][]:修復画像データ(ising)
# g[][]:受信画像データ(ising)
def res_tanaka(g, h, w):
    s = copy.copy(g)

    a = np.zeros((h, w, TA_POTS_Q))
    for i in range(h):
        for j in range(w):
            for k in range(TA_POTS_Q):
                a[i][j][k] = 1 / TA_POTS_Q

    for r in range(TA_R):
        print("開始")

        t = TA_C * math.log(2) / math.log(r + 2)
        # t = 1.0

        diff = TA_TH
        while diff >= TA_TH:
            b = np.zeros((h, w, TA_POTS_Q))
            diff = 0.0
            for i in range(h):
                for j in range(w):
                    e = np.zeros(TA_POTS_Q)
                    z = 0
                    for k in range(TA_POTS_Q):
                        sum = 0
                        if i != 0:
                            sum += a[i - 1][j][k]
                        if i != h - 1:
                            sum += a[i + 1][j][k]
                        if j != 0:
                            sum += a[i][j - 1][k]
                        if j != w - 1:
                            sum += a[i][j + 1][k]
                        e[k] = -kd(k, s[i][j]) - TA_J * sum
                        # e[k] = -diff_rate(k, s[i][j], TA_POTS_Q) - TA_J * sum

                        z += math.exp(-e[k] / t)

                    for k in range(TA_POTS_Q):
                        b[i][j][k] = math.exp(-e[k] / t) / z

                        diff += abs(b[i][j][k] - a[i][j][k])

            for i in range(h):
                for j in range(w):
                    for k in range(TA_POTS_Q):
                        a[i][j][k] = b[i][j][k]

            print(diff)

        for i in range(h):
            for j in range(w):
                max_k = 0
                max_a = 0
                for k in range(TA_POTS_Q):
                    if max_a < a[i][j][k]:
                        max_a = a[i][j][k]
                        max_k = k
                s[i][j] = max_k

        tmp_img = cv2.imread("img/Lighthouse.bmp")
        tmp_img_pots = get_img_grad(tmp_img, s, TA_POTS_Q, h, w)
        print((r + 1), "回目終了")
        cv2.waitKey(0)
        cv2.imshow("portrait", tmp_img_pots)
        print("キーを押して続行")
        cv2.waitKey(0)

    return s


def res_heikinnka(g, h, w):
    s = copy.copy(g)
    a = copy.copy(g)

    for i in range(h):
        for j in range(w):
            sum = 0
            sum += filter_additon(h, w, i - 1, j + 1, s) / 9
            sum += filter_additon(h, w, i - 1, j, s) / 9
            sum += filter_additon(h, w, i - 1, j - 1, s) / 9
            sum += filter_additon(h, w, i, j + 1, s) / 9
            sum += filter_additon(h, w, i, j, s) / 9
            sum += filter_additon(h, w, i, j - 1, s) / 9
            sum += filter_additon(h, w, i + 1, j + 1, s) / 9
            sum += filter_additon(h, w, i + 1, j, s) / 9
            sum += filter_additon(h, w, i + 1, j - 1, s) / 9

            a[i][j][0] = int(sum)
            a[i][j][1] = int(sum)
            a[i][j][2] = int(sum)

    return a


def res_gaussian(g, h, w):
    s = copy.copy(g)
    a = copy.copy(g)

    for i in range(h):
        for j in range(w):
            sum = 0
            sum += filter_additon(h, w, i - 1, j + 1, s) * 1 / 16
            sum += filter_additon(h, w, i - 1, j, s) * 2 / 16
            sum += filter_additon(h, w, i - 1, j - 1, s) * 1 / 16
            sum += filter_additon(h, w, i, j + 1, s) * 2 / 16
            sum += filter_additon(h, w, i, j, s) * 4 / 16
            sum += filter_additon(h, w, i, j - 1, s) * 2 / 16
            sum += filter_additon(h, w, i + 1, j + 1, s) * 1 / 16
            sum += filter_additon(h, w, i + 1, j, s) * 2 / 16
            sum += filter_additon(h, w, i + 1, j - 1, s) * 1 / 16

            a[i][j][0] = int(sum)
            a[i][j][1] = int(sum)
            a[i][j][2] = int(sum)

    return a


def res_median(g, h, w):
    s = copy.copy(g)
    a = copy.copy(g)

    for i in range(h):
        for j in range(w):
            val_ary = np.zeros(9)
            val_ary[0] = filter_additon(h, w, i - 1, j + 1, s)
            val_ary[1] = filter_additon(h, w, i - 1, j, s)
            val_ary[2] = filter_additon(h, w, i - 1, j - 1, s)
            val_ary[3] = filter_additon(h, w, i, j + 1, s)
            val_ary[4] = filter_additon(h, w, i, j, s)
            val_ary[5] = filter_additon(h, w, i, j - 1, s)
            val_ary[6] = filter_additon(h, w, i + 1, j + 1, s)
            val_ary[7] = filter_additon(h, w, i + 1, j, s)
            val_ary[8] = filter_additon(h, w, i + 1, j - 1, s)

            val_ary.sort()
            cent_val = val_ary[4]

            a[i][j][0] = cent_val
            a[i][j][1] = cent_val
            a[i][j][2] = cent_val

    return a


def filter_additon(h, w, i, j, s):
    if i == -1 or i == h or j == -1 or j == w:
        return 0
    else:
        return s[i][j][0]


def get_img_bin(ising, h, w):
    img_bin = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            if ising[i][j] == 1:
                val = 255
            else:
                val = 0
            img_bin[i][j][0] = val
            img_bin[i][j][1] = val
            img_bin[i][j][2] = val
    return img_bin


def get_img_grad(img_org, pots, p_grad, h, w):
    img_pots = copy.copy(img_org)
    for i in range(h):
        for j in range(w):
            diff = 256 / p_grad
            val = int((pots[i][j] + 0.5) * diff) - 1
            img_pots[i][j][0] = val
            img_pots[i][j][1] = val
            img_pots[i][j][2] = val
    return img_pots


# クロネッカーのデルタ
def kd(a, b):
    return 1 if a == b else 0


# 階調値が近いほど1に近く、遠いほど0に近くなる関数
def diff_rate(a, b, p_grad):
    return (p_grad - abs(a - b)) / p_grad
    # tmp = (p_grad - abs(a - b)) / p_grad
    # return (tmp * 2) - 1.0
