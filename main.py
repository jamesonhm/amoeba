import pygame

WIDTH = 1280
HEIGHT = 720
FPS = 60

class PetreeDish:
    def __init__(self, radius):
        self.radius = radius

        self.screen = None
        self.clock = None
        self.font = None
        
        self.reset()

    def reset(self):
        """Reset the game to initial state"""
        self.amoebas = [(WIDTH // 2, HEIGHT // 2)]
        # generate food
        self.score = 0
        self.game_over = False
        # get observation

    def take_action(self, action):
        """
        Take action in the game
        Action is either Move 1 of 4 dirs or Divide
        """
        pass

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    dt = 0

    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

    while running:
        # poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill('purple')

        pygame.draw.circle(screen, 'red', player_pos, 40)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player_pos.y -= 300 * dt
        if keys[pygame.K_s]:
            player_pos.y += 300 * dt
        if keys[pygame.K_a]:
            player_pos.x -= 300 * dt
        if keys[pygame.K_d]:
            player_pos.x += 300 * dt

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limit FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-independent physics
        dt = clock.tick(FPS) / 1000
    pygame.quit()

if __name__ == "__main__":
    main()
