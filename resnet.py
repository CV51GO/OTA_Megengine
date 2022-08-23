import megengine.functional as F
import megengine.module as M
import megengine.hub as hub
import megengine
import numpy as np


class Conv2d(M.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class FrozenBatchNorm2d(M.Module):
  
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = megengine.Parameter(megengine.tensor(F.ones(num_features), dtype=np.float32))
        self.bias = megengine.Parameter(megengine.tensor(F.zeros(num_features), dtype=np.float32))
        self.running_mean = megengine.Parameter(megengine.tensor(F.zeros(num_features), dtype=np.float32))
        self.running_var = megengine.Parameter(megengine.tensor(F.zeros(num_features) - eps, dtype=np.float32))

    def forward(self, x):
        scale = self.weight * 1/F.sqrt(self.running_var + self.eps)
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class ResNetBlockBase(M.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        # for p in self.parameters():
        #     p.requires_grad = False
        # FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

class BottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        activation=None,
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.activation = get_activation(activation)
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.activation(out)

        return out

def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            # "BN": BatchNorm2d,
            # "SyncBN": NaiveSyncBatchNorm,
            # "SyncBN1d": NaiveSyncBatchNorm1d,
            "FrozenBN": FrozenBatchNorm2d,
            # "GN": lambda channels: nn.GroupNorm(32, channels),
            # "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)



def get_activation(activation):
    """
    Args:
        activation (EasyDict or str):

    Returns:
        nn.Module or None: the activation layer
    """
    if activation is None:
        return None

    atype = activation['NAME']
    # inplace = activation.INPLACE
    act = {
        "ReLU": M.ReLU,
        # "ReLU6": M.ReLU6,
    }[atype]
    return act()

class BasicStem(M.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN", activation=None):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        # weight_init.c2_msra_fill(self.conv1)

        self.activation = get_activation(activation)
        self.max_pool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4 



def make_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks.

    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        first_stride (int): the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks


class ResNet(M.Module):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = M.Sequential(*blocks)
            name = "res" + str(i + 2)
            # self.add_module(name, stage)
            setattr(self, name, stage)

            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

       

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = F.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    # def output_shape(self):
    #     return {
    #         name: ShapeSpec(
    #             channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
    #         )
    #         for name in self._out_features
    #     }



def build_resnet_backbone():

    depth = 50
    in_channels = 3 
    out_channels = 64
    activation = {'NAME': 'ReLU', 'INPLACE': True}
    norm = 'FrozenBN'

    stem = BasicStem(in_channels, out_channels, norm=norm, activation=activation)

    # freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    # if freeze_at >= 1:
    #     for p in stem.parameters():
    #         p.requires_grad = False
    #     stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features = ['res3', 'res4', 'res5']
    num_groups = 1
    width_per_group = 64
    bottleneck_channels = num_groups * width_per_group
    in_channels = 64
    out_channels = 256
    stride_in_1x1 = True
    res5_dilation = 1
    num_classes = None
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3]}[depth]

    # Avoid creating variables without gradients
    # which consume extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5, "linear": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on Res2, Res3, Res4, Res5
  
    stages = []

    in_channels = in_channels
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
            "activation": activation,
        }

        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
    
        stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        # if freeze_at >= stage_idx:
        #     for block in blocks:
        #         block.freeze()
        stages.append(blocks)

    return ResNet(stem,
                  stages,
                  num_classes=num_classes,
                  out_features=out_features)


if __name__ == '__main__':
    resnet = build_resnet_backbone()
    resnet.eval()
    input = megengine.Tensor(np.random.randn(1, 3, 800, 1216))
    out = resnet(input)

    print('pass')
