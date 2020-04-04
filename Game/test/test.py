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
    print(env.action_space.n)
    env.close()
    return


if __name__ == "__main__":
    main()

