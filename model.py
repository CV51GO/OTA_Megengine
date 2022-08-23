import megengine.functional as F
import megengine.module as M
import megengine.hub as hub
import megengine
import numpy as np
from fpn import build_retinanet_fpn_backbone
from head import FCOSHead
from instances import Instances
from boxes import Boxes
import copy
from image_list import ImageList

def build_backbone():
    backbone = build_retinanet_fpn_backbone()
    return backbone

def permute_to_N_HWA_K(tensor, K):
    N, _, H, W = tensor.shape
    tensor = tensor.reshape(N, -1, K, H, W)
    tensor = tensor.transpose(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K) 
    return tensor

def batched_nms(boxes, scores, idxs, iou_threshold):
    assert boxes.shape[-1] == 4
    if len(boxes) < 40000:
        return F.nn.nms(boxes, scores, iou_threshold)
    else:
        raise NotImplementedError

def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())
    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)
    results = results[output_boxes.nonempty()]
    return results

def _create_grid_offsets(size, stride, offset, device):
    grid_height, grid_width = size
    shifts_start = offset * stride
    shifts_x = F.arange(shifts_start, grid_width * stride + shifts_start, step=stride, dtype=np.float32)
    shifts_y = F.arange(shifts_start, grid_height * stride + shifts_start, step=stride, dtype=np.float32)
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shift_x = megengine.tensor(shift_x).reshape(-1)
    shift_y = megengine.tensor(shift_y).reshape(-1)
    return shift_x, shift_y

class ShiftGenerator(M.Module):
    def __init__(self):
        super().__init__()
        self.num_shifts = 1
        self.strides    = [8, 16, 32, 64, 128]
        self.offset     = 0.0
        self.num_features = len(self.strides)

    @property
    def num_cell_shifts(self):
        return [self.num_shifts for _ in self.strides]

    def grid_shifts(self, grid_sizes, device):
        shifts_over_all = []
        for size, stride in zip(grid_sizes, self.strides):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, device)
            shifts = F.stack((shift_x, shift_y), axis=1)
            shifts_over_all.append(F.repeat(shifts, self.num_shifts, axis=0))
        return shifts_over_all

    def forward(self, features):
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        shifts_over_all = self.grid_shifts(grid_sizes, features[0].device)

        shifts = [copy.deepcopy(shifts_over_all) for _ in range(num_images)]
        return shifts

def build_shift_generator():
    return ShiftGenerator()

class Shift2BoxTransform(object):
    def __init__(self, weights):
        self.weights = weights

    def apply_deltas(self, deltas, shifts):
        if deltas.size == 0:
            return F.empty_like(deltas)
        deltas = deltas.reshape(deltas.shape[:-1] + (-1, 4)) / megengine.tensor(self.weights, dtype=np.float32)
        boxes = F.concat((F.expand_dims(shifts, -2) - deltas[..., :2],F.expand_dims(shifts, -2) + deltas[..., 2:]), axis=-1).reshape(deltas.shape[:-2] + (-1, ))
        return boxes

def cat(tensors, dim=0):
    if len(tensors) == 1:
        return tensors[0]
    return F.concat(tensors, dim)

class FCOS(M.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 80
        self.in_features = ['p3', 'p4', 'p5', 'p6', 'p7']
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.score_threshold = 0.05
        self.topk_candidates = 1000
        self.nms_threshold = 0.6
        self.max_detections_per_image = 100
        self.backbone = build_backbone()
        self.head = FCOSHead()
        self.shift_generator = build_shift_generator()
        self.shift2box_transform = Shift2BoxTransform(weights=[1.0, 1.0, 1.0, 1.0])
        pixel_mean = megengine.tensor([103.53, 116.28, 123.675]).reshape(3, 1, 1)
        pixel_std = megengine.tensor([1.0, 1.0, 1.0]).reshape(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_iou = self.head(features)
        shifts = self.shift_generator(features)
        results = self.inference(box_cls, box_delta, box_iou, shifts,images)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def inference(self, box_cls, box_delta, box_iou, shifts, images):
        assert len(shifts) == len(images)
        results = []
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_iou = [permute_to_N_HWA_K(x, 1) for x in box_iou]
        # list[Tensor], one per level, each has shape (N, Hi x Wi, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_iou_per_image = [
                box_iou_per_level[img_idx] for box_iou_per_level in box_iou
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, box_iou_per_image,
                shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, box_iou, shifts, image_size):
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        for box_cls_i, box_reg_i, box_iou_i, shifts_i in zip(box_cls, box_delta, box_iou, shifts):
            box_cls_i = F.sqrt(F.sigmoid(box_cls_i) * F.sigmoid(box_iou_i))
            box_cls_i = F.flatten(box_cls_i)
            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.shape[0])
            predicted_prob, topk_idxs = F.sort(box_cls_i, descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        images = [x["image"]for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def _inference_for_ms_test(self, batched_inputs):
        assert not self.training, "inference mode with training=True"
        assert len(batched_inputs) == 1, "inference image number > 1"
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)

        results = self.inference(box_cls, box_delta, box_center, shifts, images)
        for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results = detector_postprocess(results_per_image, height, width)
        return processed_results

@hub.pretrained(
"https://studio.brainpp.com/api/v1/activities/3/missions/60/files/d0ea7679-af9e-4700-b6fc-d56aa4cd33af"
)
def get_megengine_FCOS_model():
    model_megengine = FCOS()
    return model_megengine
