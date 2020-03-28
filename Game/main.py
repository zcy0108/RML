import retro
import numpy as np
import action
import Q
import CNN

# main file
# obs observation 210*160*3
# rew float reward
# done boolean
# info score 1 & 2


def Running_algorithm(env):
    # Initialize replay memory D to capacity N
    # Initialize action-value function Q with random weights

    Cases = 100000  # for each case, the game will run once times until its fail
    case = 0  # counting running cases
    time_interval = 0  # if it's large enough, network will be updated
    theta = np.array([1600, 512])
    while case < Cases:
        obs = env.reset()
        # act = action.get_greedily(env, state, theta, 0.1)
        env.render()
        while True:

            state = CNN.run(obs)
            act = action.get_greedily(env, state, theta, 0.1)

            obs, rew, done, info = env.step(act)
            # env.render()  # show the running animation
            if done:
                break
        case += 1
    #  end running cases
    return


def main():  # main function
    env = retro.make("Breakout-Atari2600")

    Running_algorithm(env)

    # close the environment
    env.close()


if __name__ == "__main__":
    main()
