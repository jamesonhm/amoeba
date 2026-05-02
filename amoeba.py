import math
import numpy as np
import pygame


class Amoeba:
    def __init__(self, start_x, start_y):
        self.radius = 30
        self.vector = pygame.math.Vector2(start_x, start_y)

        self.energy_max = 300
        self.energy = 200
        self.move_cost = 5
        self.divide_cost = 250

        self.obs_dist = 150
        self.obs_count = 20

        self._memory = None

    @property
    def position(self):
        return self.vector.x, self.vector.y

    @property
    def obs_angles(self):
        return [(math.pi * i/(self.obs_count / 2)) for i in range(self.obs_count)]

    @property
    def obs_array(self):
        angles = self.obs_angles
        x, y = self.vector.x, self.vector.y
        d = self.obs_dist
        return [((d * math.cos(angle)) + x, (d * math.sin(angle)) + y) for angle in angles]

    def store(self, prev_obs):
        """
        Save an observation for future refence
        Args:
            prev_obs(dict[string]float32): dict of previous observations
        """

        self._memory = prev_obs

    def recall(self):
        return self._memory

    def move_to(self, new_x, new_y):
        self.vector.x = new_x
        self.vector.y = new_y
        self.energy -= self.move_cost

    def eat(self, food):
        if self.energy + food.energy > self.energy_max:
            self.energy = self.energy_max
        else:
            self.energy += food.energy

    def normalize_detect(self, arr):
        """Takes an observation raycast array and normalizes to a range of 0.0 - 1.0"""
        arr = np.array(arr, dtype=np.float32)
        arr = np.where(np.isinf(arr), 1.0, arr / self.obs_dist)
        return arr

    def detect(self, others):
        """
        determines if another object can be seen and if so at what distance
        distance is to the near edge along the direction from self
        Args:
            others(list[other]): "other" has an x and y location property and a radius
        Returns:
            dists(list[float]): distances edge to edge for each observation angle
        """
        dists = []
        for dir in self.obs_angles:
            t = np.inf
            for other in others:
                # center to center dist
                wx = self.vector.x - other.vector.x
                wy = self.vector.y - other.vector.y

                dx, dy = math.cos(dir), math.sin(dir)

                dot = wx * dx + wy * dy
                disc = dot * dot - (wx * wx + wy * wy - other.radius * other.radius)
                
                if disc < 0:
                    continue                    # ray in this dir misses the object

                sqrt_disc = math.sqrt(disc)
                t_near = -dot - sqrt_disc       # smaller root - near side of obj
                t_far = -dot + sqrt_disc        # larger root - far side

                if t_far < 0:
                    continue                    # object is fully behind player

                # t_near < 0 means player is already overlapping - use t_far as exit
                t = t_near if t_near >= 0 else t_far
            dists.append(t if t <= self.obs_dist else np.inf)
        return dists

    def detect_wall(self, game):
        """
        distance from player to the edge in a given direction in radians
        uses ray-circle intersection: solves t^2 + 2(v dot d)t * (|v|^2 - R^2)  = 0
        P: player pos
        C: circle area pos
        R: radius circle area
        v: P - C
        t: dist to solve for
        d: direction of move
        t = -(v dot d) + sqrt([(v dot d)^2 - (|v|^2 - R^2)]
        """
        R = game.radius             # effective boundary radius
        dists = []
        for dir in self.obs_angles:
            dx, dy = math.cos(dir), math.sin(dir)       # unit direction vector

            # Vector from center to player
            vx = game.vector.x - self.vector.x
            vy = game.vector.y - self.vector.y

            dot = vx * dx + vy * dy                     # v dot d
            dist_sq = vx * vx + vy * vy                 # |v|^2

            discriminant = dot * dot - (dist_sq - R * R)    # always >= 0 inside circle

            # positive root = distance to the wall ahead
            dist = -dot + math.sqrt(max(0.0, discriminant))
            dists.append(dist if dist <= self.obs_dist else np.inf)
        return dists
