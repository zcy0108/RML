import retro
import numpy as np
import tensorflow as tf
import Network


def main():  # main function
    env = retro.make("Breakout-Atari2600")
    # noinspection PyArgumentList
    agent = Network.QNetwork(env)
    obs = env.reset()
    while True:
        action = agent.get_action_greedily()
        next_obs, rew, done, info = env.step(action)
        agent.store_transition(obs, rew, action, next_obs)
        if done:
            env.reset()
        if agent.end:
            break
    env.close()  # close the environment


if __name__ == "__main__":
    main()
