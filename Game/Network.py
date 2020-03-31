import random
import numpy as np
import tensorflow as tf
import test
from gym import spaces

filename = 'theta.txt'


def read(r, c):
    f = open(filename)
    lines = f.readlines()
    f.close()
    theta = np.random.rand(r, c)
    row = 0
    for line in lines:
        line = line.strip().split('\n')
        theta[row, :] = line[:]
        row += 1
    return theta


def save(theta):
    f = open(filename, 'w+')
    for i in theta:
        for j in i:
            print(j, end=' ', file=f)
        print()
    f.close()
    return


class QNetwork:
    end = False  # stop running or not
    alpha = 0.01  # learning rate
    epsilon = 0.1  # greedy rate
    gamma = 0.9  # discount rate

    action_size = 512
    state_size = 2520
    interval_size = 300  # train once time in each interval
    training_size = 50  # training example size

    database_size = 1000
    database_state = np.zeros((database_size, 164, 144, 3))
    database_reward = np.zeros(database_size)
    database_action = np.zeros((database_size, 8))
    database_next_state = np.zeros((database_size, 164, 144, 3))
    database_ite = 0

    def __int__(self):
        self.fix_theta = read(1600, 512)  # weight matrix theta for evaluation
        self.training_theta = self.fix_theta  # weight matrix theta for training
        return self

    def get_action_greedily(self):
        if np.random.rand() < self.epsilon:
            return spaces.multi_binary.MultiBinary(8).sample()
        else:
            return self.get_max_action()

    def get_max_action(self):

        return spaces.multi_binary.MultiBinary(8).sample()

    def run(self):

        return

    def transfer_observation(self, obs):
        # rows are from 32 to 196
        # columns are from 8 to 152
        state = np.zeros((164, 144, 3))
        for i in range(164):
            state[i] = obs[i+32][8:152][:]
        return state

    def train(self):
        data = random.sample(range(0, self.database_size), self.training_size)

        # ite =
        return

    def store_transition(self, obs, reward, action, next_obs):
        self.database_ite = self.database_ite % self.database_size
        self.database_state[self.database_ite] = self.transfer_observation(obs)
        self.database_reward[self.database_ite] = reward
        self.database_action[self.database_ite] = action
        self.database_next_state[self.database_ite] = self.transfer_observation(next_obs)
        self.database_ite += 1
        return
