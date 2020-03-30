import random
import numpy as np
import tensorflow as tf

filename = 'theta.txt'


def read():
    f = open(filename)
    lines = f.readlines()
    f.close()
    theta = np.zeros((2520, 512))
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
    database_state = np.zeros((database_size, 2520))
    database_reward = np.zeros(database_size)
    database_action = np.zeros((database_size, 8))
    database_next_state = np.zeros((database_size, 2520))
    database_ite = 0

    def __int__(self):
        self.fix_theta = read()  # weight matrix theta for evaluation
        self.training_theta = self.fix_theta  # weight matrix theta for training
        return self

    def get_action_greedily(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2, 8)
        else:
            return self.get_max_action()

    def get_max_action(self):
        return np.random.randint(0, 2, 8)

    def run(self):

        return

    def transfer_observation(self, obs):
        L1 = len(obs)
        L2 = len(obs[0])
        lis = list()
        for i in range(32, 93):
            if not i % 6:
                for j in range(8, L2 - 8):
                    if not j % 2:
                        k = self.to1(obs[i][j])
                        lis.append(k)
        for i in range(93, L1 - 17):
            if not i % 4:
                for j in range(8, L2 - 8):
                    if not j % 2:
                        k = self.to1(obs[i][j])
                        lis.append(k)
        return np.array(lis).reshape(1, self.state_size)

    def train(self):
        samples = random.sample(range(0, self.database_size), self.training_size)
        print(samples)
        # ite =
        return

    def to1(self, a):
        if a[0] == a[1] and a[1] == a[2] and a[2] == 0:
            return -1
        else:
            return 1

    def store_transition(self, obs, reward, action, next_obs):
        self.database_ite = self.database_ite % self.database_size
        self.database_state[self.database_ite] = self.transfer_observation(obs)
        self.database_reward[self.database_ite] = reward
        self.database_action[self.database_ite] = action
        self.database_next_state[self.database_ite] = self.transfer_observation(next_obs)
        self.database_ite += 1
        return
