import retro
import numpy as np
import tensorflow as tf
import Network

f = open("test/out.txt", 'w+')

# main file
# obs observation 210*160*3
# rew float reward
# done boolean
# info score 1 & 2


def main():  # main function
    env = retro.make("Breakout-Atari2600")
    agent = Network.QNetwork(env)
    env.reset()
    while True:
        action = agent.get_action()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()
        if agent.end:
            break
    env.close()  # close the environment


if __name__ == "__main__":
    main()
