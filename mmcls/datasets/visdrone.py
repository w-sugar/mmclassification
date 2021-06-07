from .builder import DATASETS
from .imagenet import ImageNet

@DATASETS.register_module()
class VisDrone(ImageNet):
    CLASSES = ['background', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
