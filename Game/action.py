import retro
import numpy as np
import math


def toInt(a):
    ans = 0
    for i in range(8):
        ans += math.pow(a[i], 8 - i)
    return ans


def toAry(env, a):
    ans = env.action_space.sample()
    for i in range(8):
        ans[i] = a // math.pow(2, 8 - i)
    return ans


def Get_max():

    return


def get_greedily(env, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return Get_max()
