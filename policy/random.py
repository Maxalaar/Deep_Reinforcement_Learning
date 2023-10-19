from policy.policy import Policy
from typing import Dict
import numpy as np


class Random(Policy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

    def compute_action(self, observation):
        return self._action_space.sample()

    def add_data(self, observation: Dict[str, np.ndarray], action: Dict[str, np.ndarray]):
        pass