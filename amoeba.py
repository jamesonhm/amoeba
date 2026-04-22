
import pygame


class Amoeba:
    def __init__(self, start_x, start_y):
        self.radius = 30
        self.vector = pygame.math.Vector2(start_x, start_y)
        self.energy_max = 300
        self.energy = 200
        self.move_cost = 10
        self.divide_cost = 250

    def move_to(self, new_x, new_y):
        self.vector.x = new_x
        self.vector.y = new_y
        self.energy -= self.move_cost

