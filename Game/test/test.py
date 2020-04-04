import math
import random
import retro
import time
from numba import cuda
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import Network
from gym import spaces

f = open("test/out.txt", "w+")
np.set_printoptions(threshold=np.inf)


def main():  # main function
    env = retro.make("Breakout-Atari2600")
    obs = env.reset()
    for i in range(100):
        obs, rew, done, info = env.step(env.action_space.sample())
    state = np.where(np.reshape(obs[32:196, 8:152, 0:1], (164, 144)) > 0, 1, -1)
    lis = [i for i in range(164)]
    # print(lis[0:164:2])
    state = np.delete(state, lis[0:164:2], axis=0)
    state = np.delete(state, lis[0:144:2], axis=1)
    print(np.size(state))
    env.close()
    return


if __name__ == "__main__":
    main()

