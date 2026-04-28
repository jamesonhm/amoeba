import copy
import math
import numpy as np
import pygame

from amoeba import Amoeba
from food import Food

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
        Action is either None, Move 1 of n dirs, or Divide (reproduce)
        Returns:
            observation(dict[type]array): the data from _get_obs()
            reward(int): negative for not moving, slight positive for move, 
                more positive for eating, most positive for dividing
            terminated(bool): whether an end condition has been met - player dies
            truncated(bool): stop condition due to time or number of steps
            info():?
        """
        # action in the space [1:n+1] indicate a move
        player_pos = copy.copy(self.amoebas[0].vector)
        reward = -1
        terminated = False
        truncated = False
        info = None

        if action == 0:
            return self._get_obs(), reward, terminated, truncated, info
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

            if player_pos != self.amoebas[0].vector:
                # player moved some distance
                reward += 2
            # update position
            self.amoebas[0].move_to(player_pos.x, player_pos.y)

            # check food collisions
            for food in self.food:
                if self.amoebas[0].vector.distance_squared_to(food.vector) <= (self.amoebas[0].radius + food.radius) ** 2:
                    self.food.remove(food)
                    self.amoebas[0].eat(food)
                    self._generate_food()
                    reward += 9
                    self.score += reward

        if self.amoebas[0].energy <= 0:
            terminated = True
            self.game_over = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Current state observation for RL
        observation space is a dict
            {"food": np.array[10], "enemy": np.array[10], "wall": np.array[10]}
        """
        food = self.amoebas[0].detect(self.food)
        # enemies = self.amoebas[0].detect(self.enemies)
        wall = self.amoebas[0].detect_wall(self)
        return {"food": food, "wall": wall}

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
        food_radius = 10
        food_energy = 75
        proximity = True
        while proximity:
            pos = self._random_vector_in_area()
            for amoeba in self.amoebas:
                if amoeba.vector.distance_squared_to(pos) < self._min_food_dist ** 2:
                    break
                proximity = False
            if not proximity:
                self.food.append(Food(pos.x, pos.y, food_radius, food_energy))

    def _random_vector_in_area(self):
        theta = float(np.random.uniform(0, 2*np.pi, 1)[0])
        # square root ensures uniform distribution over the area
        # r = radius * sqrt(U) where U is uniform (0, 1)
        r = self.radius * np.sqrt(float(np.random.uniform(0, 1, 1)[0]))

        # convert polar to cartesian and add to area center
        x = r * np.cos(theta) + self.vector.x
        y = r * np.sin(theta) + self.vector.y

        return pygame.math.Vector2(float(x), float(y))

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Petree Dish")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 32)

        # Clear screen
        self.screen.fill('purple')
        pygame.draw.circle(self.screen, 'black', self.vector, self.radius)

        # Draw amoeba
        for amoeba in self.amoebas:
            for pt in amoeba.obs_array:
                pygame.draw.line(self.screen, 'white', amoeba.vector, pt, 1)
            pygame.draw.circle(self.screen, 'red', amoeba.vector, amoeba.radius)

        # Draw food
        for food in self.food:
            pygame.draw.circle(self.screen, 'green', food.vector, food.radius)

        # Info Text
        text_x_start = self.radius * 2 + 50

        player_energy_text = self.font.render(f"Energy: {self.amoebas[0].energy} / {self.amoebas[0].energy_max}", True, "white")
        player_energy_loc = (text_x_start, 20)
        self.screen.blit(player_energy_text, (player_energy_loc))

        score_text = self.font.render(f"Score: {self.score}", True, "white")
        score_loc = (text_x_start, 50)
        self.screen.blit(score_text, score_loc)
        # game over display?


        # flip() the display to put your work on screen
        pygame.display.flip()
        # limit FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-independent physics
        if self.clock:
            self.clock.tick()

    def close(self):
        """Close the rendering window"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def play_manual(self, fps=10):
        """Play the game with manual controls"""
        if self.screen is None:
            self.render()

        paused = False
        running = True

        while running:
            # poll for events
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        action = 1
                    if event.key == pygame.K_s:
                        action = 3
                    if event.key == pygame.K_a:
                        action = 4
                    if event.key == pygame.K_d:
                        action = 2

            if not paused and not self.game_over:
                self.take_action(action)

            # Render
            self.render()

            self.clock.tick(fps)

        self.close()
        print(f'GAME ENDED')

def main():

    game = PetreeDish()

    try:
        game.play_manual()
    except KeyboardInterrupt:
        game.close()
        print("\nGame Interrupt")

if __name__ == "__main__":
    main()
