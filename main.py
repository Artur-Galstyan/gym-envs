from gym_envs.envs.rocket_landing import RocketLandingEnv


def main():
    env = RocketLandingEnv()
    for i in range(1000):
        env.physics_step(1.0 / 60)
        env.render()


if __name__ == "__main__":
    main()
