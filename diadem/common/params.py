# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import collections
import json
import yaml
from .anneal import Anneal, LinAnneal, ExpAnneal


class Params(collections.MutableMapping):

    def __init__(self, **kwargs):
        filename = kwargs.pop("filename", None)
        if filename:
            if isinstance(filename, str) and filename.endswith(".json"):
                self._load_from_json(filename)
            elif isinstance(filename, str) and filename.endswith(".yaml") or \
                isinstance(filename, list):
                self._load_from_yaml(filename)          
        else:
            self.store = dict()
            self.update(dict(**kwargs))

        self.param_descriptions = dict()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            new_key = key[0]
            # save the description then check if we can return a vale
            self.param_descriptions[new_key] = key[1]
            if len(key) == 3:  # annealing specification given
                if self.is_annealed(key[2]):
                    if not isinstance(self.store[new_key], Anneal):
                        if new_key in self.store:
                            annealer = self.get_annealer(
                                key[2], self.store[new_key])
                        else:
                            annealer = self.get_annealer(key[2], Params())
                        self.store[new_key] = annealer
                    return self.store[new_key]
        else:
            new_key = key
        if new_key in self.store:
            return self.store[new_key]
        else:
            self.store[new_key] = Params()
            return self.store[new_key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            new_key = key[0]
            self.param_descriptions[new_key] = key[1]
        else:
            new_key = key
        self.store[new_key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _load_from_json(self, fn):
        with open(fn) as file:
            data = json.load(file)
        self.convert_to_param(data)


    def _merge(self, source, destination):
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                self.merge(value, node)
            else:
                destination[key] = value

        return destination
    
    def _load_from_yaml(self, filename):
        """
        Merges specific parameters with common parameters
        """
        config_str = ''
        for file in filename:
            with open(file) as yaml_file:
                config_str += yaml_file.read() + '\n'
        json_parameters = yaml.load(config_str)
        #del parameters['base']
        self.convert_to_param(json_parameters)

    def convert_to_param(self, new_dict):
        self.store = dict()
        for key, value in new_dict.items():
            if isinstance(value, dict):
                param = Params()
                self.store[key] = param.convert_to_param(value)
            else:
                self.store[key] = value

        return self

    def convert_to_dict(self, print_description=False):
        dict = {}
        for key, value in self.store.items():
            if isinstance(value, Params):
                v = value.convert_to_dict(print_description)
                if len(v) == 0:
                    if print_description:
                        if key in self.param_descriptions:
                            dict[key] = self.param_descriptions[key]
                        else:
                            dict[key] = "--"
                    else:
                        dict[key] = "MISSING"
                else:
                    dict[key] = v
            elif isinstance(value, Anneal):
                if print_description:
                    dict[key] = self.param_descriptions[key]
                else:
                    v = value.params.convertToDict(print_description)
                    dict[key] = v
            else:
                dict[key] = value

                if print_description:
                    if key in self.param_descriptions:
                        dict[key] = self.param_descriptions[key]
                    else:
                        dict[key] = "--"

        return dict

    def save(self, filename, print_description=False):
        with open(filename, 'w') as outfile:
            outfile.write(json.dumps(self.convert_to_dict(
                print_description=print_description), indent=4))

    def is_annealed(self, second_key):
        return second_key.lower() == "lin" or second_key.lower() == "exp"

    def get_annealer(self, second_key, value):
        if second_key.lower() == "lin":
            return LinAnneal(params=value)
        elif second_key.lower() == "exp":
            return ExpAnneal(params=value)
        else:
            return "Unknown annealer specification."

    def anneal(self, current_step):
        for _, val in self.store.items():
            if isinstance(val, Anneal) or isinstance(val, Params):
                val.anneal(current_step=current_step)
