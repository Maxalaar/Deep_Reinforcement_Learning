from typing import Optional, List, Dict
import numpy as np
from gymnasium.core import ObsType, ActType

from algorithm.follow_policy import follow_policy
from algorithm.value_function.action_value_function import ActionValueFunction
from environment.environment import Environment
from policy.policy import Policy


class DeepQNetwork:
    def __init__(
        self,
        environment: Environment,
        policy: Policy,
        action_value_function: ActionValueFunction,
        discount_rate: float = 0.99,
    ):
        self._environment: Environment = environment
        self._policy: Policy = policy
        self._action_value_function: ActionValueFunction = action_value_function
        self._discount_rate = discount_rate

    def learning(self, number_policy_improvement_steps: int, sample_size: Optional[int], depth_temporal_difference: Optional[int]):
        for i in range(number_policy_improvement_steps):
            self._environment.reset()
            observation = self._environment.observation()
            information = self._environment.information()
            for action in range(0, self._environment.action_space.n):
                estimation_action_value = self._compute_estimation_action_value(information, action, sample_size, depth_temporal_difference)
                self._action_value_function.add_data(observation, action, estimation_action_value)

    def _compute_estimation_action_value(
            self,
            initial_state: Dict[str, np.ndarray],
            initial_action: ActType,
            sample_size: int = 1,
            depth_temporal_difference: Optional[int] = None
    ) -> float:
        sum_estimation_sample: float = 0
        for i in range(0, sample_size):
            self._environment.reset(seed=None, options={'configuration': initial_state})
            sum_estimation_sample += follow_policy(environment=self._environment, policy=self._policy, initial_action=initial_action, number_steps=depth_temporal_difference, discount_rate=self._discount_rate)

            if not self._environment.is_terminated():
                observation = self._environment.observation()
                action = self._policy.compute_action(observation)
                sum_estimation_sample += (self._discount_rate**depth_temporal_difference) * self._action_value_function.predict_value(observation, action)

        estimation_action_value = sum_estimation_sample / float(sample_size)
        return estimation_action_value



