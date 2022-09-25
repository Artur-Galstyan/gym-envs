from gym_envs.envs.mujoco_ball import MujocoBallEnv


def main():
    env = MujocoBallEnv()

    for _ in range(10000):
        env.step()
        env.render()


if __name__ == "__main__":
    main()
