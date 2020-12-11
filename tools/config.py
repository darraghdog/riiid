import json

def load_config(config_file):
    with open(config_file, "r") as fd:
        config = json.load(fd)
    _merge(defaults, config)
    return config
