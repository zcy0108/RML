import random
import retro
import time
import numpy as np
import tensorflow as tf
import Network


def cut(obs):
    state = np.zeros((164, 144, 3))
    for i in range(164):
        state[i] = obs[i + 32][8:152][:]
    return state


class Data:
    ite = 0
    size = 10000
    sample_size = 1000
    state = np.zeros((size, 164, 144, 3))
    action = np.zeros((size, 8))
    reward = np.zeros(size)
    next_state = np.zeros((size, 164, 144, 3))

    def __int__(self):
        return self

    def insert(self, state, action, reward, next_state):
        t = self.ite % self.size
        self.state[t] = cut(state)
        self.action[t] = action
        self.reward[t] = reward
        self.next_state[t] = cut(next_state)
        self.ite += 1
        return

    def get_data(self, its):
        st = np.zeros((self.sample_size, 164, 144, 3))
        act = np.zeros((self.sample_size, 8))
        rew = np.zeros(self.sample_size)
        next_st = np.zeros((self.sample_size, 164, 144, 3))
        i = 0
        for it in its:
            st[i] = self.state[it]
            act[i] = self.action[it]
            rew[i] = self.reward[it]
            next_st[i] = self.next_state[it]
            i += 1
        return self.state, self.action, self.reward, self.next_state


def make_training_data():
    env = retro.make("Breakout-Atari2600")
    obs = env.reset()
    pre_lives = 5
    d = Data()
    for i in range(20000):
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        cur_lives = info.get('lives')
        if pre_lives != cur_lives:
            pre_lives = cur_lives
            rew = -1
        d.insert(obs, action, rew, next_obs)
        obs = next_obs
    env.close()
    return d


def main():  # main function
    inp = make_training_data()
    sample = random.sample(range(0, 10000), 1000)
    state, action, reward, next_state = inp.get_data(sample)



    return


if __name__ == "__main__":
    main()

