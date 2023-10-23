import json
import pprint

from model.model import Model


def convert_data_key(data_key) -> str:
    return json.dumps(data_key.tolist())


class Tabular(Model):
    def __init__(self, default_value=0):
        super().__init__()
        self._dictionary = {}
        self._default_value = default_value

    def create(self, inpout_dimension, output_dimension):
        pass

    def add_data(self, value_input, value_output):
        self._dictionary[convert_data_key(value_input)] = value_output

    def get_data(self, value_input) -> float:
        data_key = convert_data_key(value_input)
        if data_key in self._dictionary:
            return self._dictionary[data_key][0]
        else:
            return self._default_value

    def __len__(self):
        return len(self._dictionary)

    def __str__(self):
        return pprint.pformat(self._dictionary)
