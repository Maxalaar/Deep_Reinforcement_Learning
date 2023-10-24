from algorithm.q_learning import QLearning
from algorithm.follow_policy import follow_policy
from value_function.action_value_function import ActionValueFunction
from environment.environment import Environment, space_to_size
from environment.grid_world.grid_world import GridWorld
from model.neural_networks.architecture.dense_neural_network import DenseNeuralNetwork
from model.neural_networks.neural_networks import NeuralNetworks
from model.tabular import Tabular
from policy.policy import Policy
from policy.random import Random

from policy.max_state_action_function import MaxStateActionFunction

if __name__ == '__main__':
    map_size = (4, 4)
    max_steps = 10
    render_configuration = {
        'window_size': ([element * 130 for element in map_size]),
        'fps': 5,
    }

    # Environment creation
    environment: Environment = GridWorld(
        map_size,
        max_steps,
        render_configuration,
    )
    observation_space_size = space_to_size(environment.observation_space)
    action_space_size = space_to_size(environment.action_space)

    # Neural Network Architecture creation
    architecture = DenseNeuralNetwork(
        2,
        10,
    )

    # Model creation
    tabular_model = Tabular()
    neural_networks_model = NeuralNetworks(
        architecture,
        1,
        0.01,
    )

    # Value function creation
    action_value_function: ActionValueFunction = ActionValueFunction(
        observation_space_size,
        action_space_size,
        neural_networks_model,
    )

    # Policy creation
    random_policy: Policy = Random(
        environment.observation_space,
        environment.action_space,
    )
    max_q_function_policy: Policy = MaxStateActionFunction(
        environment.observation_space,
        environment.action_space,
        action_value_function,
        0.0,
    )

    # Algorithm creation
    q_learning: QLearning = QLearning(
        environment,
        max_q_function_policy,
        action_value_function,
        0.7,
    )

    # Learning
    q_learning.learning(
        35000,
        1,
        1,
    )

    for i in range(0, 1000):
        environment.reset()
        max_q_function_policy._exploration_rate = 0
        print(follow_policy(environment, max_q_function_policy, render_environment=True, discount_rate=1))
