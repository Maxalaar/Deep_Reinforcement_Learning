from abc import ABC, abstractmethod
from typing import Dict

import gymnasium
import numpy as np


def space_dictionary_to_nparray(dictionary):
    merged_array = np.array([])
    for key in dictionary:
        merged_array = np.append(merged_array, dictionary[key])
    return merged_array


def space_to_size(space):
    input_dimension = 0
    if type(space) is gymnasium.spaces.dict.Dict:
        input_dimension += len(space_dictionary_to_nparray(space.sample()))
    elif type(space) is gymnasium.spaces.discrete.Discrete:
        input_dimension += 1
    else:
        assert 'Attention that this space is not supported.'
    return input_dimension


class Environment(gymnasium.Env, ABC):
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
