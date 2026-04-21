import copy
import math
import pygame

from amoeba import Amoeba

SIDE_LENGTH = 720
FPS = 60
MOVE_DIST = 30

class PetreeDish:
    def __init__(self):
        self.vector = pygame.math.Vector2(SIDE_LENGTH // 2, SIDE_LENGTH // 2)
        self.radius = 350

        self.screen = None
        self.clock = None
        self.font = None
        
        self.reset()

    def reset(self):
        """Reset the game to initial state"""
        self.amoebas = [Amoeba(SIDE_LENGTH // 2, SIDE_LENGTH // 2)]
        # generate food
        self.score = 0
        self.game_over = False
        # get observation

    def take_action(self, action):
        """
        Take action in the game
        Action is either None, Move 1 of 4 dirs, or Divide (reproduce)
        """
        # action in the space [1:5] indicate a move
        # calculate allowed move
        player_pos = copy.copy(self.amoebas[0].vector)
        
        if action == 0:
            return None
        elif action == 1:
            # up, 3pi/2 rad
            max_move = self._max_move(self.amoebas[0], math.pi*(3/2))
            player_pos.y -= min(MOVE_DIST, max(0.0, max_move))
        elif action == 2:
            # right, 0 rad
            max_move = self._max_move(self.amoebas[0], 0.0)
            player_pos.x += min(MOVE_DIST, max(0.0, max_move))
        elif action == 3:
            # down, pi/2 rad
            max_move = self._max_move(self.amoebas[0], math.pi/2)
            player_pos.y += min(MOVE_DIST, max(0.0, max_move))
        elif action == 4:
            # left, pi rad
            max_move = self._max_move(self.amoebas[0], math.pi)
            player_pos.x -= min(MOVE_DIST, max(0.0, max_move))

        # update position
        # if self.vector.distance_squared_to(player_pos) < (self.radius - self.amoebas[0].radius)**2:
        self.amoebas[0].move_to(player_pos.x, player_pos.y)

    def _player_angle(self, player: Amoeba):
        """
        angle from center of board circle to player position in radians
        """
        return math.atan2((player.vector.y - self.vector.y), (player.vector.x - self.vector.x))

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

def main():
    pygame.init()
    screen = pygame.display.set_mode((SIDE_LENGTH, SIDE_LENGTH))
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
            pygame.draw.circle(screen, 'red', amoeba.vector, amoeba.radius)


        # flip() the display to put your work on screen
        pygame.display.flip()

        # limit FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-independent physics
        dt = clock.tick(FPS) / 1000
    pygame.quit()

if __name__ == "__main__":
    main()
