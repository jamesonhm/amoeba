
import pygame


class Food:
    def __init__(self, pos_x, pos_y, radius, energy):
        self.vector = pygame.math.Vector2(pos_x, pos_y)
        self.radius = radius
        self.energy = energy
