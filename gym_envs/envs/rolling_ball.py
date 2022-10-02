import numpy as np
import pygame
import pymunk as mk
from pymunk.pygame_util import DrawOptions
import gym
from gym.utils.renderer import Renderer
from typing import Optional


def limit_velocity(body, gravity, damping, dt):
    max_velocity = 500
    mk.Body.update_velocity(body, gravity, damping, dt)
    l = body.velocity.length
    if l > max_velocity:
        scale = max_velocity / l
        body.velocity = body.velocity * scale
    max_angular_vel = 20
    angular_drag = 0.35

    if body.angular_velocity < 0:
        body.angular_velocity += angular_drag
    elif body.angular_velocity > 0:
        body.angular_velocity -= angular_drag
    if abs(body.angular_velocity) > max_angular_vel:
        body.angular_velocity = max_angular_vel * np.sign(body.angular_velocity)


class RollingBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = "human", human_play=False):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.human_play = human_play
        self.render_mode = render_mode
        self.window_size = 600

        # Physics Setup
        self.space = mk.Space()
        self.space.gravity = 0, 0

        wall_size = 10
        north_wall = mk.Body(body_type=mk.Body.STATIC)
        north_wall.position = self.window_size / 2, 0
        north_wall_poly = mk.Poly.create_box(
            north_wall, size=(self.window_size, wall_size)
        )
        north_wall_poly.color = (255, 125, 255, 100)

        self.space.add(north_wall, north_wall_poly)

        south_wall = mk.Body(body_type=mk.Body.STATIC)
        south_wall.position = self.window_size / 2, self.window_size
        south_wall_poly = mk.Poly.create_box(
            south_wall, size=(self.window_size, wall_size)
        )
        south_wall_poly.color = (255, 125, 255, 100)

        self.space.add(south_wall, south_wall_poly)

        east_wall = mk.Body(body_type=mk.Body.STATIC)
        east_wall.position = self.window_size, self.window_size / 2
        east_wall_poly = mk.Poly.create_box(
            east_wall, size=(wall_size, self.window_size)
        )
        east_wall_poly.color = (255, 125, 255, 100)

        self.space.add(east_wall, east_wall_poly)

        west_wall = mk.Body(body_type=mk.Body.STATIC)
        west_wall.position = 0, self.window_size / 2
        west_wall_poly = mk.Poly.create_box(
            west_wall, size=(wall_size, self.window_size)
        )
        west_wall_poly.color = (255, 125, 255, 100)

        self.space.add(west_wall, west_wall_poly)

        self.agent = mk.Body()
        self.agent.position = 300, 300
        self.agent.velocity_func = limit_velocity
        agent_poly = mk.Circle(self.agent, radius=25)
        agent_poly.mass = 5
        agent_poly.color = (125, 200, 175, 100)
        agent_poly.friction = 0.89

        self.space.add(self.agent, agent_poly)
        self.agent_force_multiplier = 1000

        if render_mode == "human":
            self.init_render_window()

        # self.renderer = Renderer(self.render_mode, self._render_frame)

    def init_render_window(self):
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.render_draw_options = DrawOptions(self.window)

    def _physics_step(self, dt):
        self.space.step(dt)

    def step(self, action):
        if action == 0:
            self.agent.angular_velocity += -5
        elif action == 1:
            self.agent.angular_velocity += 5
        elif action == 2:
            angle = self.agent.angle
            impulse = 100 * mk.Vec2d(np.cos(angle), np.sin(angle)).rotated(-angle)
            self.agent.apply_impulse_at_local_point(impulse)
        elif action == 3:
            # NO OP action
            pass
        self._physics_step(1.0 / 60)

    def render(self):
        if self.render_mode == "human":
            if self.window == None:
                self.init_render_window()

            self.window.fill("white")
            self.space.debug_draw(self.render_draw_options)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                elif self.human_play and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.step(0)
                    elif event.key == pygame.K_d:
                        self.step(1)
                    elif event.key == pygame.K_w:
                        self.step(2)
            if self.human_play:
                self._physics_step(1.0 / 60)
