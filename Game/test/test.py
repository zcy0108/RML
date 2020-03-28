import retro
import CNN


def toLarge(a):
    return a[0] * 1000000 + a[1] * 1000 + a[2]


def judge(a):
    if a[0] == a[1] and a[1] == a[2] and a[2] == 0:
        return True
    return False


def main():
    f = open("test/out.txt", 'w+')
    env = retro.make("Breakout-Atari2600")
    env.reset()

    for i in range(100):
        obs, rew, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()
        env.render()
    L1 = len(obs)
    L2 = len(obs[0])
    # obs = obs[32:][8:(L - 8)][:]

    # for i in range(32, L1 - 17):
    #     for j in range(8, L2 - 8):
    #         # print("{:0>9d}".format(toLarge(obs[i][j])), end=' ', file=f)
    #         k = 1 if not judge(obs[i][j]) else 0
    #         print(k, end=' ', file=f)
    #     print(' ', file=f)

    a = CNN.Initialize(obs)
    for i in a:
        for j in i:
            print(j, end='', file=f)
        print('', file=f)

    env.close()

    print(len(a), len(a[0]))
    return


if __name__ == "__main__":
    main()
