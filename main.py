from algorithm.deep_q_network import DeepQNetwork
from algorithm.follow_policy import follow_policy
from algorithm.value_function.action_value_function import ActionValueFunction
from environment.environment import Environment
from environment.grid_world.grid_world import GridWorld
from model.tabular import Tabular
from policy.policy import Policy
from policy.random import Random

from policy.max_state_action_function import MaxStateActionFunction

if __name__ == '__main__':
    map_size = (3, 3)
    max_steps = 10
    render_configuration = {
        'window_size': ([element * 130 for element in map_size]),
        'fps': 5,
    }

    environment: Environment = GridWorld(
        map_size,
        max_steps,
        render_configuration,
    )

    model = Tabular()
    action_value_function: ActionValueFunction = ActionValueFunction(
        model
    )

    random_policy: Policy = Random(
        environment.observation_space,
        environment.action_space,
    )

    max_q_function_policy = MaxStateActionFunction(
        environment.observation_space,
        environment.action_space,
        action_value_function,
        0.0,
    )

    deep_q_network: DeepQNetwork = DeepQNetwork(
        environment,
        max_q_function_policy,
        action_value_function,
        0.7,
    )

    deep_q_network.learning(
        1000,
        1,
        None,
    )
    print(model)

    print('len : ' + str(len(model._dictionary)))
    number_zero = 0
    for key in model._dictionary:
        if model._dictionary[key] == 0:
            number_zero += 1
    print('number_zero : ' + str(number_zero))

    for i in range(0, 1000):
        environment.reset()
        max_q_function_policy._exploration_rate = 0
        print(follow_policy(environment, max_q_function_policy, render_environment=True, discount_rate=1))

