import gymnasium
import numpy as np

from environment.environment import space_dictionary_to_nparray
from model.model import Model


class ActionValueFunction:
    def __init__(self, observation_space_size: int, action_space_size: int, model):
        self._model: Model = model
        self._model.create(observation_space_size + action_space_size, 1)

    def predict_value(self, observation, action) -> float:
        merged_dictionary = {**observation, **{'action': np.array([action])}}
        value = self._model.get_data(space_dictionary_to_nparray(merged_dictionary))
        return value

    def add_data(self, observation, action, value):
        merged_dictionary = {**observation, **{'action': np.array([action])}}
        self._model.add_data(space_dictionary_to_nparray(merged_dictionary), np.array([value]))
