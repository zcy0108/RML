import retro
import Network


def main():  # main function
    env = retro.make("Breakout-Atari2600")
    agent = Network.QNetwork()
    obs = env.reset()
    pre_lives = 5
    # Training
    Steps = 10000000
    case = 1
    for step in range(Steps):  # training episodes 10 million
        action = agent.get_action_greedily(obs)
        next_obs, rew, done, info = env.step(action)
        cur_lives = info.get('lives')
        if pre_lives != cur_lives:
            pre_lives = cur_lives
            rew = -1
        agent.store_transition(obs, rew, action, next_obs)
        obs = next_obs
        # env.render()
        if (not step % agent.interval_size) and step > agent.database_size:
            agent.train()
        if (not step % (100 * agent.interval_size)) and step > agent.database_size:
            agent.save()
        if done:
            pre_lives = 5
            print("No.", case, "The score is", info.get('score'))
            case += 1
            env.reset()
    # Testing
    obs = env.reset()
    while True:
        action = agent.get_action_greedily(obs)
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            print("The score is", info.get('score'))
            break
    env.close()  # close the environment


if __name__ == "__main__":
    main()
