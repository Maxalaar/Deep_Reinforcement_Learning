from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def add_data(self, data_key, data_value):
        pass

    @abstractmethod
    def get_data(self, data_key) -> float:
        pass