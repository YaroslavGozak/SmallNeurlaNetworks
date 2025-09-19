import sys
import random
import os
import pygame
import subprocess
import atexit

# ------------- CONFIG -------------
CELL_SIZE     = 20
GRID_WIDTH    = 30
GRID_HEIGHT   = 20
SCREEN_WIDTH  = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = 10

# Colors
BLACK = (  0,   0,   0)
GREEN = (  0, 255,   0)
RED   = (255,   0,   0)
WHITE = (255, 255, 255)

# Path to the new wallpaper you want to set
NEW_WALLPAPER = "/home/yasso/map.png"
GOAL_SCORE = 10
# ----------------------------------

def change_wallpaper():
    # --feh-- set full-screen background
    subprocess.call(["pcmanfm", "--set-wallpaper", NEW_WALLPAPER])


def place_food(snake):
    positions = [
        (x, y)
        for x in range(GRID_WIDTH)
        for y in range(GRID_HEIGHT)
        if (x, y) not in snake
    ]
    return random.choice(positions)


def main():
    pygame.init()
    pygame.joystick.init()

    # Якщо є хоча б один джойстик — ініціалізуємо його
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Detected joystick: {joystick.get_name()}")

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()

    snake     = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    direction = (1, 0)
    food      = place_food(snake)

    score = 0
    font  = pygame.font.SysFont(None, 36)  # Font, default 36

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Keyboard
            # elif event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_UP and direction != (0, 1):
            #         direction = (0, -1)
            #     elif event.key == pygame.K_DOWN and direction != (0, -1):
            #         direction = (0, 1)
            #     elif event.key == pygame.K_LEFT and direction != (1, 0):
            #         direction = (-1, 0)
            #     elif event.key == pygame.K_RIGHT and direction != (-1, 0):
                    # direction = (1, 0)

            # Gamepad: D-pad (hat) рух
            elif event.type == pygame.JOYAXISMOTION:
                x_hat, y_hat = 0, 0
                if event.axis == 0:
                    x_hat = round(event.value, 0)
                elif event.axis == 1:
                    y_hat = -round(event.value, 0)

                if x_hat == -1 and direction != (1, 0):
                    direction = (-1, 0)
                elif x_hat == 1 and direction != (-1, 0):
                    direction = (1, 0)
                if y_hat == 1 and direction != (0, 1):
                    direction = (0, -1)
                elif y_hat == -1 and direction != (0, -1):
                    direction = (0, 1)

        head_x, head_y = snake[0]
        dx, dy = direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)

        # Hit self - end game
        if new_head in snake:
            pygame.quit()
            sys.exit()

        snake.insert(0, new_head)
        if new_head == food:
            score += 1
            food = place_food(snake)
        else:
            snake.pop() # Get rid of tail

        # If snake is of length 6 - game win
        if len(snake) > GOAL_SCORE:
            change_wallpaper()
            pygame.quit()
            sys.exit()

        screen.fill(BLACK)
        for x, y in snake:
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, rect)
        fx, fy = food
        pygame.draw.rect(
            screen,
            RED,
            pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

        score_surf = font.render(f"Score: {score}/{GOAL_SCORE}", True, WHITE)
        screen.blit(score_surf, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
