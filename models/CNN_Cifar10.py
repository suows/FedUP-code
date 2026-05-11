# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel
import torchvision


class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.num_classes = config.num_classes
        self.model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes) 
        self.add_module('classifier', self.model.fc)

    def forward(self, x):
        return self.model(x)




