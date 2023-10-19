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

    def add_data(self, data_key, data_value):
        self._dictionary[convert_data_key(data_key)] = data_value

    def get_data(self, data_key) -> float:
        data_key = convert_data_key(data_key)
        if data_key in self._dictionary:
            return self._dictionary[data_key]
        else:
            return self._default_value

    def __str__(self):
        return pprint.pformat(self._dictionary)
