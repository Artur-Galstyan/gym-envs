import mujoco
import mujoco_viewer


class MujocoBallEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(
            "gym_envs/envs/models/mujoco_ball.xml"
        )
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def render(self):
        if self.viewer.is_alive:
            self.viewer.render()

    def step(self):
        mujoco.mj_step(self.model, self.data)
