import numpy as np
import retro
from gym import spaces

f = open("test/out.txt", "w+")

# rows are from 32 to 196
# columns are from 8 to 152


def main():
    env = retro.make("Breakout-Atari2600")
    obs = env.reset()
    print(spaces.multi_binary.MultiBinary(8).sample())
    env.close()
    return


if __name__ == "__main__":
    main()
