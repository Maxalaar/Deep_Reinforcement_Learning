from typing import Tuple

import numpy as np
import pygame
from pygame import Color

from environment.environment import Environment


class GraphicInterface:
    def __init__(self, environment, render_configuration):
        self._environment: Environment = environment
        self._screen = pygame.display.set_mode(
            (render_configuration['window_size'][0], render_configuration['window_size'][1])
        )
        self._render_fps = render_configuration['fps']
        self._background_color = Color(255, 255, 255, 255)
        self._agent_color: Color = Color(255, 0, 0, 255)
        self._target_color: Color = Color(255, 255, 0, 255)
        self._clock = pygame.time.Clock()

    def _draw_rectangle(self, size, position, color: Color):
        surface = pygame.Surface(size)
        surface.set_alpha(color.a)
        surface.fill(color)
        self._screen.blit(surface, position)

    def update(self):
        self._screen.fill(self._background_color)
        square_size: Tuple[float, float] = (self._screen.get_size()[0] / self._environment.information()['map_size'][0], self._screen.get_size()[1] / self._environment.information()['map_size'][1])

        self._draw_rectangle(
            square_size,
            np.array(self._environment.information()['agent_position']) * square_size,
            self._agent_color
        )
        self._draw_rectangle(
            square_size,
            np.array(self._environment.information()['target_position']) * square_size,
            self._target_color
        )

        pygame.display.flip()
        self._clock.tick(self._render_fps)
