import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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
    nonsense = False
    alpha = 0.01  # learning rate
    epsilon = 0.1  # greedy rate
    gamma = 0.9  # discount rate

    left = [1, 0, 0, 0, 0, 0, 1, 0]
    right = [1, 0, 0, 0, 0, 0, 0, 1]

    ite = [i for i in range(164)]

    interval_size = 100  # train once time in each interval
    training_size = 32  # training example size
    database_size = 10000
    database_state = np.zeros((database_size, 5904), dtype="float32")
    database_reward = np.zeros(database_size, dtype="float32")
    database_action = np.zeros(database_size, dtype="int32")  # left is 0, right is 1
    database_next_state = np.zeros((database_size, 5904), dtype="float32")
    database_ite = 0

    model = tf.keras.Sequential()

    def __int__(self):
        self.model.add(layers.Dense(769, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(layers.Dense(96, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(layers.Dense(2))
        return self

    def get_action_greedily(self, obs):
        if np.random.rand() < self.epsilon:
            if np.random.rand() < 0.5:
                return self.left
            else:
                return self.right
        else:
            return self.get_max_action(obs)

    def get_max_action(self, obs):
        state = self.simplify(np.where(np.reshape(obs[32:196, 8:152, 0:1], (164, 144)) > 0, 1., -1.))
        judge = self.model.predict(state, batch_size=32)
        if judge[0][0] > judge[0][1]:
            return self.left
        else:
            return self.right

    def set_eps(self, eps):
        self.epsilon = eps
        return

    def simplify(self, state):
        state = np.delete(state, self.ite[0:164:2], axis=0)
        state = np.delete(state, self.ite[0:144:2], axis=1)
        return state.reshape(1, 5904)

    def store_transition(self, obs, reward, action, next_obs):
        self.database_ite = self.database_ite % self.database_size
        self.database_state[self.database_ite] = self.simplify(np.where(np.reshape(obs[32:196, 8:152, 0:1], (164, 144)) > 0, 1., -1.))
        self.database_reward[self.database_ite] = reward
        self.database_action[self.database_ite] = 0 if action[6] == 1 else 1
        self.database_next_state[self.database_ite] = self.simplify(np.where(np.reshape(next_obs[32:196, 8:152, 0:1], (164, 144)) > 0, 1., -1.))
        self.database_ite += 1
        return

    def train(self):
        data_ite = random.sample(range(0, self.database_size), self.training_size)
        data_st = np.zeros((self.training_size, 5904), dtype="float32")
        data_rw = np.zeros(self.training_size, dtype="float32")
        data_ac = np.zeros(self.training_size, dtype="int32")
        data_nst = np.zeros((self.training_size, 5904), dtype="float32")
        i = 0
        for it in data_ite:
            data_st[i] = self.database_state[it]
            data_rw[i] = self.database_reward[it]
            data_ac[i] = self.database_action[it]
            data_nst[i] = self.database_next_state[it]
            i += 1
        label_q = self.model.predict(data_st, batch_size=32)
        next_q = self.model.predict(data_nst, batch_size=32)
        for i in range(self.training_size):
            label_q[i][data_ac[i]] = data_rw[i] + self.gamma * max(next_q[i][0], next_q[i][1])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                           loss='mse',  # mean squared error
                           metrics=['mae'])  # mean absolute error
        self.model.fit(data_st, label_q, epochs=10, batch_size=32)
        return


