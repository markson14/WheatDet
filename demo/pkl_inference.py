import os
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def main():
    backbone = resnet_fpn_backbone('resnext101_32x8d', pretrained=False)
    backbone.out_channels = 256
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    model.load_state_dict(torch.load("../models/20200516-FPN/model_final.pth")['model'])

    print("done")
if __name__ == "__main__":
    main()
