import copy
import pygame

from amoeba import Amoeba

SIDE_LENGTH = 720
FPS = 60
MOVE_DIST = 30

class PetreeDish:
    def __init__(self):
        self.vector = pygame.Vector2(SIDE_LENGTH // 2, SIDE_LENGTH // 2)
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
        max_dist = self.radius - self.amoebas[0].radius
        player_pos = copy.copy(self.amoebas[0].vector)
        
        if action == 0:
            return None
        elif action == 1:
            # up
            player_pos.y -= MOVE_DIST
        elif action == 2:
            # right
            player_pos.x += MOVE_DIST
        elif action == 3:
            # down
            player_pos.y += MOVE_DIST
        elif action == 4:
            # left
            player_pos.x -= MOVE_DIST

        
        # update position
        if self.vector.distance_squared_to(player_pos) < (self.radius - self.amoebas[0].radius)**2:
            self.amoebas[0].move_to(player_pos.x, player_pos.y)

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
