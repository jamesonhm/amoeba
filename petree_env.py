import math
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
                "wall": spaces.Box(low=0, high=math.inf, shape=(10,), dtype=np.float32)
            }
        )

        
