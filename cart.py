from math import sin, cos, tan, pow, sqrt, radians, pi
from datetime import datetime, timedelta

class Physics:
    gravity = 9.8
    force_mag = 10.0
    tau = 0.15 # seconds between state updates

class Cart:
    def __init__(self, physics):
        self.physics = physics
        self.mass_cart = 1.0
        self.mass_pole = 0.3
        self.mass = self.mass_cart + self.mass_pole
        self.length = 100 # actually half the pole length
        self.pole_mass_length = self.mass_pole * self.length

        self.state = (0.0,0.0,0.0,0.0)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 6 * pi / 360
        # Max lifetime
        self.time_treshlod = 6

        self.action_space = 2
        self.observation_space = 4

        self.reward = 0
        self.time = datetime.now()

    def reset(self):
        self.reward = 0
        self.time = datetime.now()
        self.state = (0.0,0.0,0.0,0.0)
        return self.state

    def step(self, action):
        state_tuple = self.state
        x, x_dot, theta, theta_dot = state_tuple

        terminated = bool(
            theta < - self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or datetime.now() - self.time > timedelta(seconds=self.time_treshlod)
        )

        if terminated:
            return state_tuple, self.reward, True
        self.reward += 1

        # calculate force based on action
        force = self.physics.force_mag if action > 0 else (-1 * self.physics.force_mag)
        costheta = cos(theta)
        sintheta = sin(theta)

        temp = (
            force + self.pole_mass_length * theta_dot**2 * sintheta
        ) / self.mass
        thetaacc = (self.physics.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.mass)
        )
        xacc = temp - self.pole_mass_length * thetaacc * costheta / self.mass

        x = x + self.physics.tau * x_dot
        x_dot = x_dot + self.physics.tau * xacc
        theta = theta + self.physics.tau * theta_dot
        theta_dot = theta_dot + self.physics.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        return self.state, self.reward, False
    
    def close(self):
        print("close")
        '''
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
        '''

    def render(self):
        print("render")
        '''
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption("Pole Balancing Game")
        clock = pygame.time.Clock()

        # Cart
        pygame.draw.rect(screen, (0,0,0), (platform_x, platform_y, 80, 40), 100)

        # Pole
        pos0 = (platform_x+80//2, platform_y)
        pos1 = (
            pos0[0] + cart.length * sin(state_tuple[2]),
            pos0[1] - cart.length * cos(state_tuple[2])
        )

        pygame.draw.line(screen, BLACK, pos0, pos1, 10)
        '''