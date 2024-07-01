import math
import copy
import numpy as np

# 定数
#J = 0.4
J = 0.50
T_MAX = 2
T_MIN = 1
T_DELTA = 0.1
Q = 0.1
K = math.log((1 - Q) / Q) / 2
#THRESHOLD = 0.000001
THRESHOLD = 1.0


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
            #print(diff)
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
