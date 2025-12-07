
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights 

class FasterRCNN_RESNET50_FPN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model = fasterrcnn_resnet50_fpn(weights_backbone=ResNet50_Weights.IMAGENET1K_V2)
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)

    def forward(self, images, targets):
        return self.model(images, targets)
    
def get_model(device):
    model = FasterRCNN_RESNET50_FPN()
    return model.to(device)

