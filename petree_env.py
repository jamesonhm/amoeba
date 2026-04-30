import math
from os import terminal_size
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from main import PetreeDish

class PetreeDishEnv(gym.Env):
    """Gymnasium environment wrapper for Petree Dish - Amoeba game"""
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Initialize game
        self.game = PetreeDish()

        # Action Space
        self.action_space = spaces.Discrete(5)

        # Observation Space
        self.observation_space = spaces.Dict(
            {
                "food": spaces.Box(low=0, high=math.inf, shape=(10,), dtype=np.float32),
                "wall": spaces.Box(low=0, high=math.inf, shape=(10,), dtype=np.float32),
            }
        )

        if self.render_mode == "human":
            pygame.init()

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        observation = self.game.reset()
        info = self.game.get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """Execute one step in the environment"""
        observation, reward, terminated, truncated, info = self.game.take_action(action)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # handle pygame events to prevent the window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            self.game.render(mode="human")

    def close(self):
        self.game.close()

