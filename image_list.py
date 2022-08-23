import megengine.functional as F

class ImageList(object):
    def __init__(self, tensor, image_sizes):
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx):
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]  

    def to(self, *args, **kwargs):
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @staticmethod
    def from_tensors(tensors, size_divisibility, pad_ref_long=False, pad_value=0.0):
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
        max_size = list(max(s) for s in zip(*[img.shape for img in tensors]))
        if pad_ref_long:
            max_size_max = max(max_size[-2:])
            max_size[-2:] = [max_size_max] * 2
        max_size = tuple(max_size)

        if size_divisibility > 0:
            import math
            stride = size_divisibility
            max_size = list(max_size) 
            max_size[-2] = int(math.ceil(max_size[-2] / stride) * stride)  
            max_size[-1] = int(math.ceil(max_size[-1] / stride) * stride)  
            max_size = tuple(max_size)

        image_sizes = [im.shape[-2:] for im in tensors]

        if len(tensors) == 1:
            image_size = image_sizes[0]
            padding_size = [[0,0],[0, max_size[-2] - image_size[0]], [0, max_size[-1] - image_size[1]]]
            if all(x == 0 for x in padding_size):  # https://github.com/pytorch/pytorch/issues/31734
                batched_imgs = tensors[0].unsqueeze(0)
            else:
                padded = F.pad(tensors[0], padding_size, constant_value=pad_value)
                batched_imgs = F.expand_dims(padded, axis=0)
        else:
            batch_shape = (len(tensors),) + max_size
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        return ImageList(batched_imgs, image_sizes)
