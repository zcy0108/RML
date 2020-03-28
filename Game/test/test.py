import retro


def toLarge(a):
    return a[0] * 1000000 + a[1] * 1000 + a[2]


def judge(a):
    if a[0] == a[1] and a[1] == a[2] and a[2] == 142:
        return True
    return False


def main():
    f = open("test/out.txt", 'w+')
    env = retro.make("Breakout-Atari2600")
    env.reset()

    for i in range(100):
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()

    L = len(obs[0])
    obs = obs[32:][8:L - 8][:]

    print(len(obs))
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            print("{:0>9d}".format(toLarge(obs[i][j])), end=' ', file=f)
            # k = 1 if judge(obs[i][j]) else 0
            # print(k, end=' ', file=f)
        print(' ', file=f)

    # env.close()
    return


if __name__ == "__main__":
    main()
