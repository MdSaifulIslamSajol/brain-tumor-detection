#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 01:45:30 2024

@author: saiful
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:02:58 2024

@author: saiful
"""
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models

num_classes = 3

def ViTForImageClassification():
    print("== ViTForImageClassification ==")

    from transformers import ViTForImageClassification, ViTFeatureExtractor
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model_ft = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    model_ft.fc = nn.Sequential(
                    nn.Linear(768, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 2))
    
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)
    
    return model_ft


def ConvNextV2ForImageClassification():
    print("== ConvNextV2ForImageClassification ==")

    from transformers import  ConvNextV2ForImageClassification
    model_ft = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)
    
    return model_ft

def Swinv2ForImageClassification():
    print("== Swinv2ForImageClassification ==")

    from transformers import Swinv2ForImageClassification
    model_ft = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)
    
    return model_ft

def ImageGPTForImageClassification():
    print("== ImageGPTForImageClassification ==")

    from transformers import  ImageGPTForImageClassification
    model_ft = ImageGPTForImageClassification.from_pretrained("openai/imagegpt-small")
    num_in_features = model_ft.score.in_features  
    model_ft.classifier = nn.Linear(num_in_features ,  num_classes)
    
    return model_ft

def CvtForImageClassification():
    print("== CvtForImageClassification ==")

    from transformers import  CvtForImageClassification
    model_ft = CvtForImageClassification.from_pretrained("microsoft/cvt-13")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

    return model_ft

def EfficientFormerForImageClassification():
    print("== EfficientFormerForImageClassification ==")

    from transformers import  EfficientFormerForImageClassification
    model_ft = EfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)

    return model_ft

def PvtV2ForImageClassification():
    print("== PvtV2ForImageClassification ==")
    from transformers import  PvtV2ForImageClassification

    model_ft = PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)

    return model_ft

def MobileViTV2ForImageClassification():
    print("== MobileViTV2ForImageClassification ==")
    from transformers import  MobileViTV2ForImageClassification

    model_ft = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)
    return model_ft




import torch
from torch import nn
from torchvision import models


def resnet50ForImageClassification():
    print("== resnet50ForImageClassification ==")

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def vgg16ForImageClassification():
    print("== vgg16ForImageClassification ==")

    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

def mobilenetForImageClassification():
    print("== mobilenetForImageClassification ==")
    
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def googlenetForImageClassification():
    print("== googlenetForImageClassification ==")
    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def efficientnet_b0ForImageClassification():
    print("== efficientnet_b0ForImageClassification ==")

    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


