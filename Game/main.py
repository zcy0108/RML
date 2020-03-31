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
    agent = Network.QNetwork()
    obs = env.reset()
    pre_lives = 5
    for step in range(1000000):  # running steps
        action = re_action(env.action_space.sample(), agent.get_action_greedily())
        next_obs, rew, done, info = env.step(action)
        cur_lives = info.get('lives')
        if pre_lives != cur_lives:
            pre_lives = cur_lives
            rew = -1
        agent.store_transition(obs, rew, action, next_obs)
        obs = next_obs
        env.render()
        if (not step % agent.interval_size) and step > agent.database_size:
            agent.train()
        if done:
            pre_lives = 5
            env.reset()
    env.close()  # close the environment


if __name__ == "__main__":
    main()
