import megengine.functional as F
import megengine.module as M
import megengine

class Scale(M.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = megengine.Parameter(megengine.tensor([init_value]))

    def forward(self, input):
        return input * self.scale

class FCOSHead(M.Module):
    def __init__(self):
        super().__init__()
        in_channels = 256
        num_classes = 80
        num_convs = 4
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.norm_reg_targets = True
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                M.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(M.GroupNorm(32, in_channels))
            cls_subnet.append(M.ReLU())
            bbox_subnet.append(
                M.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(M.GroupNorm(32, in_channels))
            bbox_subnet.append(M.ReLU())

        self.cls_subnet = M.Sequential(*cls_subnet)
        self.bbox_subnet = M.Sequential(*bbox_subnet)
        self.cls_score = M.Conv2d(in_channels,
                                   num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = M.Conv2d(in_channels,
                                   4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.ious_pred = M.Conv2d(in_channels,
                                   1,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        self.scales = [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))]

    def forward(self, features):
        logits = []
        bbox_reg = []
        ious_pred = []
        for level, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)
            logits.append(self.cls_score(cls_subnet))
            ious_pred.append(self.ious_pred(bbox_subnet))
            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * self.fpn_strides[level])
        return logits, bbox_reg, ious_pred