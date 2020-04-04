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
    Steps = 10000000
    for step in range(Steps):  # training episodes 10 million
        agent.set_eps(1.0 - step/Steps)
        action = agent.get_action_greedily(obs)
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
            print("The score is", info.get('score'))
            env.reset()
    # Testing
    obs = env.reset()
    while True:
        action = agent.get_action_greedily(obs)
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            print("The score is", info.get('score'))
            break
    env.close()  # close the environment


if __name__ == "__main__":
    main()
