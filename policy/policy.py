from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from gymnasium.spaces import space


class Policy(ABC):
    def __init__(self, observation_space, action_space):
        self._observation_space: space.Space = observation_space
        self._action_space: space.Space = action_space

    @abstractmethod
    def compute_action(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass
