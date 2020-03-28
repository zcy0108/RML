import numpy as np


def Initialize(a):  # cut out the processing area
    L = len(a[0])
    return a[32:][8:L - 8][:]


def convolution():
    return


def pooling():
    return


def run(obs):
    area = Initialize(obs)
    return
