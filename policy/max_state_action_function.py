import numpy as np

from gymnasium.core import ActType

from policy.policy import Policy
from typing import Dict, Optional
import numpy as np


class MaxStateActionFunction(Policy):
    def __init__(self, observation_space, action_space, action_value_function, exploration_rate: float = 0):
        super().__init__(observation_space, action_space)
        self._action_value_function = action_value_function
        self._exploration_rate: float = exploration_rate

    def compute_action(self, observation):
        best_action: ActType = None
        best_action_value: Optional[int] = None

        if np.random.rand() <= self._exploration_rate:
            return self._action_space.sample()

        for action in range(0, self._action_space.n):
            state_action_value = self._action_value_function.predict_value(observation, action)
            if best_action_value is None or state_action_value > best_action_value:
                best_action_value = state_action_value
                best_action = action
        return best_action

    def add_data(self, observation: Dict[str, np.ndarray], action: Dict[str, np.ndarray]):
        pass
