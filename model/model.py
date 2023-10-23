from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create(self, inpout_dimension, output_dimension):
        pass

    @abstractmethod
    def add_data(self, value_input, value_output):
        pass

    @abstractmethod
    def get_data(self, value_input) -> float:
        pass
