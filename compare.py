import sys
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import megengine.functional as F
import megengine
import torch
from model import get_megengine_FCOS_model
sys.path.append('./official_OTA/cvpods')
sys.path.append('./official_OTA/OTA/playground/detection/coco/ota.res50.fpn.coco.800size.1x')
megengine.set_log_level('ERROR', update_existing=True)
class ResizeTransform():
    def __init__(self, h, w, new_h, new_w, interp):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret
    
    def __call__(self, image):
        image = self.apply_image(image)
        return image

    def _set_attributes(self, params: list = None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

class TransformGen:
    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def get_transform(self, img, annotations=None):
        raise NotImplementedError

    def __call__(self, img):
        return self.get_transform(img)(img)

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

class ResizeShortestEdge(TransformGen):
    def __init__(
        self,
        short_edge_length,
        max_size=None,
        sample_style="range",
        interp=Image.BILINEAR,
    ):
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)

# 图片路径
img_root = './data'
imgname_list = os.listdir(img_root)

image_dict_megengine = dict()
image_dict_torch = dict()

# 官方模型初始化
from config import config
from net import build_model
from cvpods.checkpoint import DefaultCheckpointer

config.merge_from_list(['MODEL.WEIGHTS', './ota.res50.fpn.coco.1x.pth'])
config.MODEL.WEIGHTS = './official_OTA/OTA/ota.res50.fpn.coco.1x.pth'
torch_model = build_model(config)
torch_model.eval()
DefaultCheckpointer(torch_model).resume_or_load(config.MODEL.WEIGHTS, resume=False)

# Megengine模型初始化
megengine_model = get_megengine_FCOS_model(pretrained=True)
megengine_model.eval()


for imgname in imgname_list:
    print(f'inference {imgname}')
    # 1.读取图片
    img_path = str(Path(img_root)/imgname)
    img = Image.open(img_path)
    image = img.convert('RGB')
    image = np.asarray(image)

    # 2.预处理
    image_dict_megengine['height'] = image.shape[0]
    image_dict_megengine['width'] = image.shape[1]
    image_dict_torch['height'] = image.shape[0]
    image_dict_torch['width'] = image.shape[1]

    image = image[:, :, ::-1]
    resize_transform = ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333, sample_style='choice')
    image = resize_transform(image)

    image_dict_megengine['image'] = megengine.tensor(F.transpose(image, (2,0,1)))
    image_dict_torch['image'] = torch.tensor(image.transpose(2, 0, 1))


    # 3.官方模型推理
    torch_start_time = time.time()
    with torch.no_grad():
        torch_outputs = torch_model([image_dict_torch])
    torch_end_time = time.time()
    print("torch inference time: {:.3f}s".format(torch_end_time-torch_start_time))

    # 4.Megengine模型推理
    megengine_start_time = time.time()
    megengine_outputs = megengine_model([image_dict_megengine])
    megengine_end_time = time.time()
    print("megengine inference time: {:.3f}s".format(megengine_end_time-megengine_start_time))

    # 5.比较推理结果
    np.testing.assert_allclose(torch_outputs[0]['instances'].pred_boxes.tensor.cpu().numpy(), megengine_outputs[0]['instances'].pred_boxes.tensor.numpy(), rtol=1e-3)
    np.testing.assert_allclose(torch_outputs[0]['instances'].scores.cpu().numpy(), megengine_outputs[0]['instances'].scores.numpy(), rtol=1e-3)
    np.testing.assert_allclose(torch_outputs[0]['instances'].pred_classes.cpu().numpy(), megengine_outputs[0]['instances'].pred_classes.numpy(), rtol=1e-3)

    print('pass')