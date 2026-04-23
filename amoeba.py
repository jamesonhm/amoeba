import math
import pygame


class Amoeba:
    def __init__(self, start_x, start_y):
        self.radius = 30
        self.vector = pygame.math.Vector2(start_x, start_y)

        self.energy_max = 300
        self.energy = 200
        self.move_cost = 10
        self.divide_cost = 250

        self.obs_dist = 30
        self.obs_count = 10

    @property
    def position(self):
        return self.vector.x, self.vector.y

    @property
    def obs_angles(self):
        return [(math.pi * i/(self.obs_count / 2)) for i in range(self.obs_count * 2)]

    @property
    def obs_array(self):
        angles = self.obs_angles
        x, y = self.vector.x, self.vector.y
        d = self.radius + self.obs_dist
        return [((d * math.cos(angle)) + x, (d * math.sin(angle)) + y) for angle in angles]

    def move_to(self, new_x, new_y):
        self.vector.x = new_x
        self.vector.y = new_y
        self.energy -= self.move_cost

    def detect(self, obj, dir):
        """
        determines if another object can be seen and if so at what distance
        distance is to the near edge along the direction from self
        
        Returns:
            hit(bool): other is along this direction, and within the obs_distance
            dist(float): distance edge to edge
        """
        # center to center dist
        wx = self.vector.x - obj.vector.x
        wy = self.vector.y - obj.vector.y
        if wx ** 2 + wy ** 2 > self.obs_dist ** 2:
            return False, math.inf
        r = self.radius + obj.radius
        dx, dy = math.cos(dir), math.sin(dir)


