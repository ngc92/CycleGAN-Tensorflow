from collections import namedtuple
import json


ImageTensors = namedtuple("ImageTensors", ("a", "b", "ab", "ba", "aba", "bab"))
LossTensors = namedtuple("LossTensors", ("G_ab", "G_ba", "D_a", "D_b", "cycle_aba", "cycle_bab"))


def _get_name(entry):
    import tensorflow as tf
    if isinstance(entry, str):
        return entry
    elif isinstance(entry, (tf.Tensor, tf.Operation)):
        return entry.name
    else:
        assert False, type(entry)


def _get_names(data_dict: dict):
    return {key: _get_name(value) for key, value in data_dict.items()}


class CycleGanModelDef:
    def __init__(self, image_tensors: ImageTensors, loss_tensors: LossTensors):
        self.image_tensors = image_tensors
        self.loss_tensors = loss_tensors

    def to_json(self, target):
        if isinstance(target, str):
            with open(target, "w") as target_file:
                return self.to_json(target_file)

        data = {
            "image_tensors": _get_names(self.image_tensors._asdict()),
            "loss_tensors": _get_names(self.loss_tensors._asdict())
        }

        json.dump(data, target, indent=2)

    @staticmethod
    def from_json(source):
        if isinstance(source, str):
            with open(source, "r") as source_file:
                return CycleGanModelDef.from_json(source_file)

        data = json.load(source)
        image_tensors = ImageTensors(**data["image_tensors"])
        loss_tensors = ImageTensors(**data["loss_tensors"])
        return CycleGanModelDef(image_tensors, loss_tensors)
