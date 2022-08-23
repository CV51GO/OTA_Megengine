import megengine.functional as F
import numpy as np

def cat(tensors, dim=0):
    if len(tensors) == 1:
        return tensors[0]
    return F.concat(tensors, dim)

class Boxes:
    def __init__(self, tensor):
        if tensor.size == 0:
            tensor = F.zeros((0, 4), dtype=np.float32)
        self.tensor = tensor

    def area(self):
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size):
        h, w = box_size
        self.tensor[:, 0] = F.clip(self.tensor[:, 0], lower=0, upper=w)
        self.tensor[:, 1] = F.clip(self.tensor[:, 1], lower=0, upper=h)
        self.tensor[:, 2] = F.clip(self.tensor[:, 2], lower=0, upper=w)
        self.tensor[:, 3] = F.clip(self.tensor[:, 3], lower=0, upper=h)

    def nonempty(self, threshold: int = 0):
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item):
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def scale(self, scale_x: float, scale_y: float):
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    def __iter__(self):
        yield from self.tensor

