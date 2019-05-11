# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import inspect
from diadem.common import Params


def extract_params(object, params):
    signature = inspect.getfullargspec(object.__init__)

    missing = False
    for idx, arg in enumerate(signature.args):
        if "self" in arg:
            continue
        if idx > len(signature.args) - len(signature.defaults):
            default_idx = idx - (len(signature.args) - len(signature.defaults))
            if isinstance(params[arg], Params):
                params[arg, "Default: " +
                       str(signature.defaults[default_idx])] = signature.defaults[default_idx]
        else:
            if arg.lower() == "MISSING":
                missing = True
            dummy = params[arg]

    dict = params.convert_to_dict()
    if not missing:
        return object(**dict)
    else:
        return None
