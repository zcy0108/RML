import retro
import numpy as np
import tensorflow as tf
import Network


def main():  # main function
    env = retro.make("Breakout-Atari2600")
    agent = Network.QNetwork()
    obs = env.reset()
    pre_lives = 5
    # Training
    for step in range(10000000):  # training episodes 10 million
        action = agent.get_action_greedily()
        next_obs, rew, done, info = env.step(action)
        cur_lives = info.get('lives')
        if pre_lives != cur_lives:
            pre_lives = cur_lives
            rew = -1
        agent.store_transition(obs, rew, action, next_obs)
        obs = next_obs
        # env.render()
        if (not step % agent.interval_size) and step > agent.database_size:
            agent.train()
        if done:
            pre_lives = 5
            env.reset()
        if not step % 1000 and step:
            print(step, "episodes done.")
    # Testing
    env.reset()
    while True:
        action = agent.get_action_greedily()
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            print("The score is", info.get('score'))
            break
    env.close()  # close the environment


if __name__ == "__main__":
    main()
