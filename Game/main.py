import gym
import retro


def main():
    env = retro.make("Breakout-Atari2600")
    Cases = 10000
    case = 0
    time_interval = 0
    while case < Cases:
        obs = env.reset()
        while True:
            k = env.action_space.sample()
            # print(k)
            obs, rew, done, info = env.step(k)
            if rew == 1:
                print(rew)
            # obs observation 210*160*3
            # rew float reward
            # done boolean
            # info score 1 & 2
            env.render()
            if done:
                break
        case += 1
    env.close()


if __name__ == "__main__":
    main()
