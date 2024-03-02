import pygame
import sys
from math import sin, cos, tan, pow, sqrt, radians, pi
from cart import Cart, Physics

from datetime import datetime, timedelta



def Game():
    pygame.init()

    WIDTH, HEIGHT = 600, 400
    PLATFORM_WIDTH, PLATFORM_HEIGHT = 80, 40
    FPS = 60

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pole Balancing Game")

    clock = pygame.time.Clock()

    platform_x = (WIDTH - PLATFORM_WIDTH) // 2
    platform_y = 300
    platform_speed = 5 

    x, x_dot, theta, theta_dot = 0.0, 0.0, 0.0, 0.0
    state_tuple = (x, x_dot, theta, theta_dot)
    physics = Physics()
    cart = Cart(physics)
    action = 0


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            action = 0
            platform_x -= platform_speed
        if keys[pygame.K_RIGHT]:
            action = 1
            platform_x += platform_speed

        
        state_tuple, reward, terminated = cart.step(action)
        if terminated:
            pygame.quit()
            sys.exit()
        print(reward)

        screen.fill(WHITE)

        pygame.draw.rect(screen, BLACK, (platform_x, platform_y, PLATFORM_WIDTH, PLATFORM_HEIGHT), 100)

        pos0 = (platform_x+PLATFORM_WIDTH//2, platform_y)
        pos1 = (
            pos0[0] + cart.length * sin(state_tuple[2]),
            pos0[1] - cart.length * cos(state_tuple[2])
        )

        pygame.draw.line(screen, BLACK, pos0, pos1, 10)

        pygame.display.flip()

        clock.tick(FPS)

    