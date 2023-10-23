from typing import Optional

import numpy as np

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim

from model.neural_networks.experience_database import ExperienceDatabase
from model.model import Model


class NeuralNetworks(Model):
    def __init__(self, architecture, data_size_for_fit=1, learning_rate=0.01):
        super().__init__()
        self._data_size_for_fit = data_size_for_fit
        self._learning_rate = learning_rate
        self._experience_database: Optional[ExperienceDatabase] = None
        self._neural_networks: nn.Module = architecture

    def create(self, inpout_dimension, output_dimension):
        self._neural_networks.create(inpout_dimension, output_dimension)
        self._experience_database = ExperienceDatabase(inpout_dimension, output_dimension)

    def add_data(self, value_input, value_output):
        self._experience_database.add_data(value_input, value_output)
        if len(self._experience_database) >= self._data_size_for_fit:
            self._fit()

    def get_data(self, value_input) -> float:
        predict_value = self._neural_networks.forward(torch.Tensor(value_input)).tolist()[0]
        return predict_value

    def _fit(self):
        self._optimizer = optim.SGD(self._neural_networks.parameters(), lr=self._learning_rate)
        self._loss_function = nn.MSELoss()

        outputs = self._neural_networks.forward(torch.Tensor(self._experience_database._data))
        loss = self._loss_function(outputs, torch.Tensor(self._experience_database._targets))
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._experience_database.clear()

