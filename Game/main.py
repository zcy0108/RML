import retro
import numpy as np
import tensorflow as tf
import Network


def re_action(a, b):
    for i in range(8):
        a[i] = b[i]
    return a


def main():  # main function
    env = retro.make("Breakout-Atari2600")
    # noinspection PyArgumentList
    agent = Network.QNetwork()
    obs = env.reset()
    while True:
        # print(env.action_space.sample())
        action = re_action(env.action_space.sample(), agent.get_action_greedily())
        next_obs, rew, done, info = env.step(action)
        agent.store_transition(obs, rew, action, next_obs)
        obs = next_obs
        if done:
            env.reset()
        if agent.end:
            break
    env.close()  # close the environment


if __name__ == "__main__":
    main()
