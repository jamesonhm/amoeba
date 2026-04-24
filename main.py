import copy
import math
import numpy as np
import pygame

from amoeba import Amoeba

WIDTH = 1280
HEIGHT = 720
CENTER_X = 360
CENTER_Y = 360
FPS = 60
MOVE_DIST = 30

class PetreeDish:
    def __init__(self):
        self.vector = pygame.math.Vector2(CENTER_X, CENTER_Y)
        self.radius = 350

        self.food_radius = 10
        self._min_food_dist = 75

        self.screen = None
        self.clock = None
        self.font = None
        
        self.reset()

    def reset(self):
        """Reset the game to initial state"""
        self.amoebas = [Amoeba(CENTER_X, CENTER_Y)]
        # generate food
        self.food = []
        self._generate_food()

        self.score = 0
        self.game_over = False
        # get observation
        return self._get_obs()

    def take_action(self, action):
        """
        Take action in the game
        Action is either None, Move 1 of n dirs, Scan, or Divide (reproduce)
        """
        # action in the space [1:n+1] indicate a move
        player_pos = copy.copy(self.amoebas[0].vector)
        
        if action == 0:
            return None
        if action in (1, 2, 3, 4):
            if action == 1:
                # up, 3pi/2 rad
                max_move = self._max_move(self.amoebas[0], math.pi*(3/2))
                player_pos.y -= min(MOVE_DIST, max_move)
            elif action == 2:
                # right, 0 rad
                max_move = self._max_move(self.amoebas[0], 0.0)
                player_pos.x += min(MOVE_DIST,  max_move)
            elif action == 3:
                # down, pi/2 rad
                max_move = self._max_move(self.amoebas[0], math.pi/2)
                player_pos.y += min(MOVE_DIST, max_move)
            elif action == 4:
                # left, pi rad
                max_move = self._max_move(self.amoebas[0], math.pi)
                player_pos.x -= min(MOVE_DIST, max_move)

            # update position
            self.amoebas[0].move_to(player_pos.x, player_pos.y)

            # check food collisions
            for food in self.food:
                if self.amoebas[0].vector.distance_squared_to(food) <= (self.amoebas[0].radius + self.food_radius) ** 2:
                    self.food.remove(food)
                    self._generate_food()

        obs = self._get_obs()

    def _get_obs(self):
        """
        Current state observation for RL
        observation space is a dict
            {"food": np.array[10], "enemy": np.array[10], "wall": np.array[10]}
        """
        food = self.amoebas[0].detect(self.food, [self.food_radius] * len(self.food))
        # enemies = self.amoebas[0].detect(self.enemies)
        # wall = self.amoebas[0].detect_wall()
        return {"food": food}

    def _max_move(self, player: Amoeba, dir):
        """
        distance from player to the edge in a given direction in radians
        uses ray-circle intersection: solves t^2 + 2(v dot d)t * (|v|^2 - R^2)  = 0
        P: player pos
        C: circle area pos
        R: radius circle area - player radius (effective radius)
        v: P - C
        t: dist to solve for
        d: direction of move
        t = -(v dot d) + sqrt([(v dot d)^2 - (|v|^2 - R^2)]
        """
        R = self.radius - player.radius             # effective boundary radius

        dx, dy = math.cos(dir), math.sin(dir)       # unit direction vector

        # Vector from center to player
        vx = player.vector.x - self.vector.x
        vy = player.vector.y - self.vector.y

        dot = vx * dx + vy * dy                     # v dot d
        dist_sq = vx * vx + vy * vy                 # |v|^2

        discriminant = dot * dot - (dist_sq - R * R)    # always >= 0 inside circle

        # positive root = distance to the wall ahead
        return -dot + math.sqrt(max(0.0, discriminant))

    def _generate_food(self):
        """
        generate food within the area and at a min distance from any players
        """
        proximity = True
        while proximity:
            food = self._random_vector_in_area()
            for amoeba in self.amoebas:
                if amoeba.vector.distance_squared_to(food) < self._min_food_dist ** 2:
                    break
                proximity = False
            if not proximity:
                self.food.append(food)

    def _random_vector_in_area(self):
        theta = float(np.random.uniform(0, 2*np.pi, 1)[0])
        # square root ensures uniform distribution over the area
        # r = radius * sqrt(U) where U is uniform (0, 1)
        r = self.radius * np.sqrt(float(np.random.uniform(0, 1, 1)[0]))

        # convert polar to cartesian and add to area center
        x = r * np.cos(theta) + self.vector.x
        y = r * np.sin(theta) + self.vector.y

        return pygame.math.Vector2(float(x), float(y))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    dt = 0

    game = PetreeDish()

    while running:
        # poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    game.take_action(1)
                if event.key == pygame.K_s:
                    game.take_action(3)
                if event.key == pygame.K_a:
                    game.take_action(4)
                if event.key == pygame.K_d:
                    game.take_action(2)


        screen.fill('purple')
        pygame.draw.circle(screen, 'black', game.vector, game.radius)

        for amoeba in game.amoebas:
            for pt in amoeba.obs_array:
                pygame.draw.line(screen, 'white', amoeba.vector, pt, 1)
            pygame.draw.circle(screen, 'red', amoeba.vector, amoeba.radius)

        for food in game.food:
            pygame.draw.circle(screen, 'green', food, game.food_radius)


        # flip() the display to put your work on screen
        pygame.display.flip()

        # limit FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-independent physics
        dt = clock.tick(FPS) / 1000
    pygame.quit()

if __name__ == "__main__":
    main()
