from collections import namedtuple


ImageTensors = namedtuple("ImageTensors", ("a", "b", "ab", "ba", "aba", "bab"))
LossTensors = namedtuple("LossTensors", ("G_ab", "G_ba", "D_a", "D_b", "cycle_aba", "cycle_bab"))


class CylceGanModelDef:
    def __init__(self, image_tensors: ImageTensors, loss_tensors: LossTensors):
        self.image_tensors = image_tensors
        self.loss_tensors = loss_tensors
