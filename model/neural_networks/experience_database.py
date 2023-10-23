import torch
import numpy as np
from torch.utils.data import Dataset


class ExperienceDatabase(Dataset):
    def __init__(self, input_dimension: int, output_dimension: int):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._data = None
        self._targets = None
        self.clear()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        x = self._data[index]
        y = self._targets[index]
        return x, y

    def add_data(self, new_data, new_targets):
        self._data = np.concatenate((self._data, np.array([new_data])), axis=0, dtype=np.float32)
        self._targets = np.concatenate((self._targets, np.array([new_targets])), axis=0, dtype=np.float32)

    def clear(self):
        self._data = np.empty((0, self._input_dimension), dtype=np.float32)
        self._targets = np.empty((0, self._output_dimension), dtype=np.float32)
