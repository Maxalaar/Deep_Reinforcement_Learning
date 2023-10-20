from typing import Optional

from gymnasium.core import ActType

from environment.environment import Environment
from policy.policy import Policy


def follow_policy(environment: Environment, policy: Policy, initial_action: Optional[ActType] = None, number_steps: Optional[int] = None, render_environment=False, discount_rate: float = 0.99) -> float:
    sum_reward: float = 0
    continue_follow_policy = True
    current_step = 0

    if initial_action is not None:
        action = initial_action
    else:
        observation = environment.observation()
        action = policy.compute_action(observation)

    while continue_follow_policy:
        if render_environment:
            environment.render()
        observation, reward, terminated, _, information = environment.step(action)
        sum_reward += (discount_rate ** current_step) * reward
        current_step += 1
        action = policy.compute_action(observation)

        if terminated or (number_steps is not None and current_step >= number_steps):
            if render_environment:
                environment.render()
            continue_follow_policy = False

    return sum_reward
