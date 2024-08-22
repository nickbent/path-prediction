import json

def json_dumps_tuple_keys(mapping):
    string_keys = {json.dumps(k): v for k, v in mapping.items()}
    return json.dumps(string_keys)

def json_loads_tuple_keys(string):
    mapping = json.loads(string)
    return {tuple(json.loads(k)): v for k, v in mapping.items()}

def load_dict(path, load_tuples = True):
    with open(path, "r") as f:
        d = json.load(f)
    if load_tuples:
        d = json_loads_tuple_keys(d)
    return d

def is_numbers(inputString):
    return all(char.isdigit() for char in inputString)