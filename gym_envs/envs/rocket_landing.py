import numpy as np
import pygame
import pymunk as mk
from pymunk.pygame_util import DrawOptions
import gym
from typing import Optional
from gym.utils.renderer import Renderer


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = "human"):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 600

        # Physics Setup
        self.space = mk.Space()
        self.space.gravity = 0, 50

        self.ground = mk.Body(body_type=mk.Body.STATIC)
        self.ground.position = self.window_size / 2, self.window_size
        self.ground_poly = mk.Poly.create_box(self.ground, (700, 25))
        self.ground_poly.color = (2, 150, 19, 100)

        self.rocket = mk.Body(1, 1000)
        self.rocket.position = 300, 300
        self.rocket_poly = mk.Poly.create_box(self.rocket, size=(100, 100))

        self.landing_spot = mk.Body(body_type=mk.Body.STATIC)
        self.landing_spot.position = (
            np.random.randint(100, self.window_size - 100),
            self.window_size - 15,
        )
        print(self.landing_spot.position)
        self.landing_spot_poly = mk.Poly.create_box(self.landing_spot, size=(100, 5))
        self.landing_spot_poly.color = (255, 115, 140, 100)

        self.space.add(self.rocket, self.rocket_poly)
        self.space.add(self.ground, self.ground_poly)
        self.space.add(self.landing_spot, self.landing_spot_poly)
        if render_mode == "human":
            self.init_render_window()

        # self.renderer = Renderer(self.render_mode, self._render_frame)

    def init_render_window(self):
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.render_draw_options = DrawOptions(self.window)

    def physics_step(self, dt):
        self.space.step(dt)

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
