import numpy as np


def to10(a):
    if a[0] == a[1] and a[1] == a[2] and a[2] == 0:
        return 0
    else:
        return 1


def Initialize(obs):  # cut out the processing area
    L1 = len(obs)
    L2 = len(obs[0])
    lis = list()
    for i in range(32, 93):
        if not i % 6:
            for j in range(8, L2 - 8):
                if not j % 2:
                    k = to10(obs[i][j])
                    lis.append(k)
    for i in range(93, L1 - 17):
        if not i % 4:
            for j in range(8, L2 - 8):
                if not j % 2:
                    k = to10(obs[i][j])
                    lis.append(k)
    return np.array(lis)


def convolution():
    return


def pooling():
    return


def run(obs):
    area = Initialize(obs)
    return area.reshape(1, 2520)
