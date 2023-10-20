import numpy as np

from model.model import Model


def dictionary_to_nparray(dictionary):
    merged_array = np.array([])
    for key in dictionary:
        merged_array = np.append(merged_array, dictionary[key])
    return merged_array


class ActionValueFunction:
    def __init__(self, model):
        self._model: Model = model

    def predict_value(self, observation, action) -> float:
        merged_dictionary = {**observation, **{'action': np.array([action])}}
        value = self._model.get_data(dictionary_to_nparray(merged_dictionary))
        return value

    def add_data(self, observation, action, value):
        merged_dictionary = {**observation, **{'action': np.array([action])}}
        self._model.add_data(dictionary_to_nparray(merged_dictionary), value)
