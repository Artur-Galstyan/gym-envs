from gym_envs.envs.mujoco_ball import MujocoBallEnv
from gym_envs.envs.rolling_ball import RollingBallEnv


def main():
    env = RollingBallEnv(human_play=True)

    for _ in range(10000):
        env.render()


if __name__ == "__main__":
    main()
