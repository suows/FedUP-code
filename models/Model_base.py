#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
import torchvision
from peft import LoraConfig, get_peft_model
import timm
from einops import rearrange
from transformers.modeling_outputs import SequenceClassifierOutput
# from utils import *
def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1./ (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    @staticmethod
    def split_weight_name(name):
        if 'weight' or 'bias' in name:
            return ''.join(name.split('.')[:-1])
        return name

    def save_params(self):
        for param_name, param in self.named_parameters():
            if 'alpha' in param_name or 'beta' in param_name:
                continue
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name, param.data.clone())

    def compute_diff(self):
        diff_mean = dict()
        for param_name, param in self.named_parameters():
            layer_name = self.split_weight_name(param_name)
            _buff_param_name = param_name.replace('.', '__')
            old_param = getattr(self, _buff_param_name, default=0.0)
            diff = (param - old_param) ** 2
            diff = diff.sum()
            total_num = reduce(lambda x, y: x*y, param.shape)
            diff /= total_num
            diff_mean[layer_name] = diff
        return diff_mean

    def remove_grad(self, name=''):
        for param_name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = False

class Adapter(nn.Module):
    def __init__(self, args, input_dim, bias=False):
        super(Adapter, self).__init__()
        self.ad = nn.Sequential(
            nn.Linear(input_dim, args.out_dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(args.out_dim, input_dim, bias=bias),
            nn.ReLU(inplace=True)
        )

        
    def forward(self, x):
        x = self.ad(x)
        return x
        

    def print_memory(self):
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        
        memory_bytes = total_params * 4  
        if memory_bytes >= 1024 ** 3: 
            size_str = f"{memory_bytes / (1024 ** 3):.2f} GB"
        elif memory_bytes >= 1024 ** 2: 
            size_str = f"{memory_bytes / (1024 ** 2):.2f} MB"
        elif memory_bytes >= 1024: 
            size_str = f"{memory_bytes / 1024:.2f} KB"
        else:
            size_str = f"{memory_bytes} Bytes"
        
        print(f"Adapter Memory: {total_params} Parameter, {size_str}")

class Lora(nn.Module):
    def __init__(self, args, base_model):
        super(Lora, self).__init__()
        self.args = args
        base_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))
        for param in base_model.parameters():
            param.requires_grad = False        
        
        if args.data_name in ['mnist', 'fashionmnist']:
            self.adapters = nn.ModuleDict({
                'adapter': Adapter(args, input_dim=512)
                
            })

            self.adapters['adapter'].print_memory()
        elif args.data_name in ['cifar10', 'cifar100']:
            self.adapters = nn.ModuleDict({
                'adapter': Adapter(args, input_dim=512)
            })
            self.adapters['adapter'].print_memory()
        elif args.data_name == 'adult':
            self.adapters = nn.ModuleDict({
                    'adapter': Adapter(args, input_dim=512)
                })
            self.adapters['adapter'].print_memory()
        elif args.data_name == 'text':
            self.adapters = nn.ModuleDict({
                'encoder.layer.11.output.dense': Adapter(input_dim=base_model.encoder.layer[11].output.dense.out_features)
            })
            self.adapters['encoder.layer.11.output.dense'].print_memory()
        else:
            raise ValueError(f"Unsupported data name: {args.data_name}")

        self.base_model = base_model
        print(self.base_model)

    def forward(self, x):
        if self.args.model == 'LeNet_FashionMNIST':
            x = self.base_model.conv1(x)    
            x = self.base_model.relu(x)     
            x = self.base_model.maxpool1(x)  
            x = self.base_model.conv2(x)    
            x = self.base_model.maxpool2(x)
            x = x.view(x.size(0), -1)            
            x = self.base_model.fc3(x)       
            x = F.relu(x)                    
            x = self.base_model.fc2(x)     
            x = F.relu(x)                       
            x = self.adapters['adapter'](x)  
            output = self.base_model.fc(x)
            return output

        elif self.args.model == 'ViT_Cifar10':
            x = self.base_model.to_patch_embedding(x)
            pe = posemb_sincos_2d(x)
            x = rearrange(x, 'b ... d -> b (...) d') + pe
            x = self.base_model.transformer(x)
            x = x.mean(dim=1)
            x = self.adapters['adapter'](x)
            output = self.base_model.linear_head(x)
            return output

        elif self.args.data_name in ['cifar10', 'cifar100']:
            if hasattr(self.base_model, 'model'):
                x = self.base_model.model.conv1(x)
                x = self.base_model.model.bn1(x)
                x = self.base_model.model.relu(x)
                x = self.base_model.model.maxpool(x)
                x = self.base_model.model.layer1(x)
                x = self.base_model.model.layer2(x)
                x = self.base_model.model.layer3(x)
                x = self.base_model.model.layer4(x)
                x = self.base_model.model.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.adapters['adapter'](x)
                output = self.base_model.model.fc(x)
                return output
            else:
                raise AttributeError(f"Model {self.args.model} for {self.args.data_name} does not have expected structure")

        else:
            raise NotImplementedError(f"Model {self.args.model} not supported in Lora forward")


class Loratext(nn.Module):
    def __init__(self, args, global_model):
        super(Loratext, self).__init__()
        self.args = args
        

        global_model.load_state_dict(
            torch.load('save_model/global_model_{}.pth'.format(args.data_name))
        )
        self.base_model = global_model
        hidden_size = global_model.model.config.hidden_size
        num_classes = global_model.model.config.num_labels

        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.adapters = nn.ModuleDict({
            'adapter': Adapter(args, input_dim=hidden_size)
        })
        self.adapters['adapter'].print_memory()
        self.classifier = self.base_model.model.classifier

    def forward(self, input_ids=None, attention_mask=None, x=None, **kwargs):
        if x is not None:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
        features = self.base_model.get_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        adapted_features = self.adapters['adapter'](features)
        logits = self.classifier(adapted_features)

        return SequenceClassifierOutput(logits=logits)
