import torch
import torch.nn as nn


# class Regressor(pl.LightningModule):
#     def __init__(self, input_size, output_size, n_layers, n_neurons):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#
#         layers = []
#         for i in range(n_layers):
#             in_features = input_size if i == 0 else n_neurons
#             out_features = n_neurons
#             layers.append(nn.Linear(in_features, out_features))
#             layers.append(nn.ReLU())
#
#         self.hidden_layers = nn.Sequential(*layers)
#         self.output_layer = nn.Linear(n_neurons, output_size)
#
#     def forward(self, x):
#         x = self.hidden_layers(x)
#         x = self.output_layer(x)
#         return x
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_predicts = self(x)
#         loss = nn.MSELoss()(y_predicts, y)
#         self.log('train_loss', loss)
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = optim.SGD(self.parameters(), lr=1e-2)
#         return optimizer

class DenseNeuralNetwork(nn.Module):
    def __init__(self, number_layer, layer_size, *args, **kwargs):
        super(DenseNeuralNetwork, self).__init__()
        self._number_layer = number_layer
        self._layer_size = layer_size

    def create(self, input_size, output_size):
        self.fc1 = nn.Linear(input_size, self._layer_size)
        self.fc2 = nn.Linear(self._layer_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x
