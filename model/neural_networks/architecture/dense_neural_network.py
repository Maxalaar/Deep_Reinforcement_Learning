import torch
import torch.nn as nn


class DenseNeuralNetwork(nn.Module):
    def __init__(self, number_layer, layer_size, *args, **kwargs):
        super(DenseNeuralNetwork, self).__init__()
        self._number_layer = number_layer
        self._layer_size = layer_size

        self.hidden_layers = None
        self.output_layer = None

    def create(self, input_size, output_size):
        layers = []
        for i in range(self._number_layer):
            in_features = input_size if i == 0 else self._layer_size
            out_features = self._layer_size
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU(0.1))

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(self._layer_size, output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
