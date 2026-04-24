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

        self.obs_dist = 90
        self.obs_count = 20

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

    def move_to(self, new_x, new_y):
        self.vector.x = new_x
        self.vector.y = new_y
        self.energy -= self.move_cost

    def detect(self, obj_locs, obj_rads):
        """
        determines if another object can be seen and if so at what distance
        distance is to the near edge along the direction from self
        Args:
            obj_locs(list[vector2]): locations of items to detect
            obj_rads(list[float]): radii of objects
        Returns:
            dists(list[float]): distances edge to edge for each observation angle
        """
        dists = []
        for dir in self.obs_angles:
            t = math.inf
            for obj, rad in zip(obj_locs, obj_rads):
                # center to center dist
                wx = self.vector.x - obj.x
                wy = self.vector.y - obj.y

                dx, dy = math.cos(dir), math.sin(dir)

                dot = wx * dx + wy * dy
                disc = dot * dot - (wx * wx + wy * wy - rad * rad)
                
                if disc < 0:
                    continue                    # ray in this dir misses the object

                sqrt_disc = math.sqrt(disc)
                t_near = -dot - sqrt_disc       # smaller root - near side of obj
                t_far = -dot + sqrt_disc        # larger root - far side

                if t_far < 0:
                    continue                    # object is fully behind player

                # t_near < 0 means player is already overlapping - use t_far as exit
                t = t_near if t_near >= 0 else t_far
            dists.append(t if t <= self.obs_dist else math.inf)
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
            dists.append(dist if dist <= self.obs_dist else math.inf)
        return dists
