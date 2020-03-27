import retro
import numpy as np
import action

# main file
# obs observation 210*160*3
# rew float reward
# done boolean
# info score 1 & 2


def Running_algorithm(env):
    # Initialize replay memory D to capacity N
    # Initialize action-value function Q with random weights

    Cases = 10000  # for each case, the game will run once times until its fail
    case = 0  # counting running cases
    time_interval = 0  # if it's large enough, network will be updated
    while case < Cases:
        obs = env.reset()
        while True:
            obs, rew, done, info = env.step()

            act = action.get_greedily(env, state, theta, 0.1)


            env.render()
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
