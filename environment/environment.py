from abc import ABC, abstractmethod
from typing import Dict

import gym
import numpy as np


class Environment(gym.Env, ABC):
    def __init__(self):
        super().__init__()
        super(ABC, self).__init__()

    @abstractmethod
    def information(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass

    @abstractmethod
    def observation(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def _reward(self) -> float:
        pass