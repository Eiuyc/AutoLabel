# -*- coding: utf-8 -*-
# file: fine_tune_model.py
# author: JinTian
# time: 10/05/2017 9:54 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
from torchvision import models
from torch import nn
IMAGE_SIZE = 224
USE_GPU = False


def resnet18_model():
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    # fine tune we change original fc layer into classes num of our own
    model_ft.fc = nn.Linear(num_features, 6)
    model_name="ResNet18"
    if USE_GPU:
        model_ft = model_ft.cuda()
    return model_ft,model_name

def resnet101_model():
    model_ft = models.resnet101(pretrained=True)
    num_features = model_ft.fc.in_features
    # fine tune we change original fc layer into classes num of our own
    model_ft.fc = nn.Linear(num_features, 6)

    model_name="ResNet101"
    if USE_GPU:
        model_ft = model_ft.cuda()
    return model_ft,model_name

def denseNet161_model():
    model_ft = models.densenet161(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    # fine tune we change original fc layer into classes num of our own
    model_ft.classifier = nn.Linear(num_ftrs, 6)
    model_name="denseNet161"
    if USE_GPU:
        model_ft = model_ft.cuda()
    return model_ft,model_name

def denseNet201_model():
    model_ft = models.densenet201(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    # fine tune we change original fc layer into classes num of our own
    model_ft.classifier = nn.Linear(num_ftrs, 6)
    model_name="denseNet201"
    if USE_GPU:
        model_ft = model_ft.cuda()
    return model_ft,model_name

def squeezenet1_1_model():
    model_ft = models.squeezenet1_1(pretrained=True)
    # num_features = model_ft.fc.in_features
    model_ft.classifier[1] = nn.Conv2d(512, 6, kernel_size=(1,1), stride=(1,1))
    # fine tune we change original fc layer into classes num of our own
    # model_ft.fc = nn.Linear(num_features, 6)
    model_name="squeezenet1_1"
    if USE_GPU:
        model_ft = model_ft.cuda()
    return model_ft,model_name

def mobileNetV3_model():
    model_ft = models.mobilenet_v3_small(pretrained=True)
    num_features = model_ft.classifier[3].in_features
    model_ft.classifier[3] = nn.Linear(num_features, 6,bias=True)
    model_name="mobileNetV3"
    if USE_GPU:
        model_ft = model_ft.cuda()
    return model_ft,model_name

