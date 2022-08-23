## 介绍

本项目是论文《OTA: Optimal Transport Assignment for Object Detection》的Megengine实现。该论文的官方实现地址：https://github.com/Megvii-BaseDetection/OTA


## 环境安装

依赖于CUDA10

```
conda create -n OTA python=3.7
pip install -r requirements.txt
```

下载官方的权重：https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EVo55E_uiHJNvtOCoMPmh5wBR0yxZs1ycIugIWTVyLIgvg?e=uIhwBs
，将下载后的文件置于./official_OTA/OTA路径下。

## 使用方法

安装完环境后，直接运行`python compare.py`。

`compare.py`文件对官方实现和Megengine实现的推理结果进行了对比。

运行`compare.py`时，会读取`./data`中存放的图片进行推理。`compare.py`中实现了Megengine框架和官方使用的Pytorch框架的推理，并判断两者推理结果的一致性。

## 模型加载示例

在使用模型时，使用如下代码即可加载模型和权重：
```python
import megengine.hub as hub
megengine_model = hub.load('CV51GO/OTA_Megengine','get_megengine_FCOS_model',pretrained=True)
```