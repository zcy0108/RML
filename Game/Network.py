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
    def __int__(self, e):
        self.env = e  # environment
        self.end = False  # stop running or not

        self.alpha = 0.01  # learning rate
        self.epsilon = 0.1  # greedy rate
        self.gamma = 0.9  # discount rate

        self.action_size = 512
        self.state_size = 2520
        self.database_size = 1000
        self.interval_size = 300  # train once time in each interval
        self.training_size = 50  # training example size

        self.fix_theta = read()  # weight matrix theta for evaluation
        self.training_theta = self.fix_theta  # weight matrix theta for training

        return self

    def get_action_greedily(self):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_max_action()

    def get_max_action(self):
        a = self.env.action_space.sample()
        return a

    def run(self):

        return

    def store_transition(self, obs, reward, action, next_obs):

        return
