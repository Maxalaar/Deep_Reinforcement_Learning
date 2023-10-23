from abc import ABC
from typing import Dict, Optional, Tuple
from gymnasium import spaces

import numpy as np
import random

from gymnasium.core import ActType

from environment.grid_world.graphic_interface import GraphicInterface
from environment.environment import Environment


class GridWorld(Environment):
    def __init__(self, map_size: Tuple[int, int], max_steps: int, render_configuration: Dict = None):
        super().__init__()

        self._map_size: Tuple[int, int] = map_size
        self._minimum_value_bounds: np.ndarray = np.array([0, 0])
        self._maximum_value_bounds: np.ndarray = np.array([self._map_size[0] - 1, self._map_size[1] - 1])

        self._max_steps: int = max_steps
        self._target_position: Optional[np.ndarray] = None
        self._agent_position: Optional[np.ndarray] = None
        self._current_step: Optional[int] = None

        self._graphic_interface: Optional[GraphicInterface] = None
        self._render_configuration: Optional[Dict] = render_configuration

        self._action_to_direction: Dict[int, np.ndarray] = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            # 4: np.array([0, 0]),
        }
        self.action_space: spaces.Discrete = spaces.Discrete(len(self._action_to_direction))

        self.observation_space: spaces.Dict = spaces.Dict(
            {
                'agent_position': spaces.Box(self._minimum_value_bounds, self._maximum_value_bounds, shape=(2,), dtype=int),
                'target_position': spaces.Box(self._minimum_value_bounds, self._maximum_value_bounds, shape=(2,), dtype=int),
            }
        )

        self.reset()

    def information(self):
        return {
            'map_size': np.array(list(self._map_size)),
            'distance_agent_target': np.array([np.linalg.norm(self._agent_position - self._target_position, ord=1)]),
            'agent_position': self._agent_position,
            'target_position': self._target_position,
            'current_step': np.array([self._current_step]),
        }

    def _set_information(self, information: Dict[str, np.ndarray]):
        self._current_step = information['current_step'][0]
        self._agent_position = information['agent_position']
        self._target_position = information['target_position']

    def _set_initialisation(self):
        self._current_step = 0
        self._agent_position = np.random.randint(self._minimum_value_bounds, self._maximum_value_bounds+1, size=2)
        self._target_position = np.random.randint(self._minimum_value_bounds, self._maximum_value_bounds+1, size=2)

    def observation(self):
        return {
            'agent_position': self._agent_position,
            'target_position': self._target_position,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and 'configuration' in options:
            self._set_information(options['configuration'])
        else:
            self._set_initialisation()

        observation = self.observation()
        information = self.information()

        return observation, information

    def _reward(self):
        reward = 0
        if np.array_equal(self._agent_position, self._target_position):
            reward += 1
        return reward

    def is_terminated(self) -> bool:
        return np.array_equal(self._agent_position, self._target_position) or self._current_step >= self._max_steps

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_position = np.clip(self._agent_position + direction, self._minimum_value_bounds, self._maximum_value_bounds)

        observation = self.observation()
        reward = self._reward()
        terminated = self.is_terminated()
        information = self.information()
        self._current_step += 1

        return observation, reward, terminated, False, information

    def render(self):
        if self._render_configuration is not None:
            if self._graphic_interface is None:
                self._graphic_interface = GraphicInterface(self, self._render_configuration)

            self._graphic_interface.update()
