import json
import numpy as np

def save_model(model, path, config=None):
    params = model.parameters()
    save_dict = {}

    for i, param in enumerate(params):
        save_dict[f"param_{i}"] = param.value

    if config is not None:
        save_dict["__config__"] = np.array(json.dumps(config), dtype=object)

    np.savez(path, **save_dict)

def load_model(model, path):
    data = np.load(path, allow_pickle=True)

    for i, param in enumerate(model.parameters()):
        param.value = data[f"param_{i}"]

def load_config(path):
    data = np.load(path, allow_pickle=True)

    if "__config__" in data:
        return json.loads(str(data["__config__"]))
    else:
        raise None