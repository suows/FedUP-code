from pickletools import optimize
import time
import math
import pandas as pd
import torch
import os
import csv
import json
from models.Model_base import *
from models import LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100
from utils import init_network, test_class_forget, test_client_forget, model_init
from dataset.data_utils import *
from algs.fl_base import Base
import torch.optim as optim
import copy
import logging
# import objgraph
import matplotlib.pyplot as plt
from utils import *
from models.Model_base import *
import torchvision.transforms as transforms
import open_clip
from collections import defaultdict
from sklearn.cluster import KMeans
import random
import pickle
import torch.nn as nn
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch

def average_sample(proto):
    proto = torch.mean(proto, dim=1, keepdim=True)
    return proto

def random_sample(proto, sample_ratio=0.1):
    sample_num = math.ceil(proto.shape[1] * sample_ratio)
    sample_idx = random.sample(range(proto.shape[1]), sample_num)
    proto = proto[:, sample_idx]
    return proto

def cluster_sample(proto, sample_ratio=0.1):
    original_device = proto.device
    original_dtype = proto.dtype
    num_samples = proto.shape[1]
    cluster_num = math.ceil(num_samples * sample_ratio)
    if num_samples == 0:
        return proto.to(torch.float32)
    if num_samples == 1 or cluster_num <= 1:
        center = proto.mean(dim=1, keepdim=True)
        return center.to(device=original_device, dtype=torch.float32)
    cluster_num = min(cluster_num, num_samples)
    compute_device = original_device
    if compute_device.type != "cuda" and torch.cuda.is_available():
        compute_device = torch.device("cuda")
    x = proto.transpose(0, 1).contiguous().to(compute_device, dtype=torch.float32)
    n_init = 5
    max_iter = 100
    tol = 1e-4

    best_inertia = None
    best_centers = None
    generator = torch.Generator(device=compute_device)
    generator.manual_seed(0)

    def pairwise_sq_dist(a, b):
        a2 = (a * a).sum(dim=1, keepdim=True)     
        b2 = (b * b).sum(dim=1).unsqueeze(0)    
        ab = a @ b.t()                             
        dist = a2 + b2 - 2 * ab
        return torch.clamp(dist, min=0.0)

    for _ in range(n_init):
        init_idx = torch.randperm(num_samples, generator=generator, device=compute_device)[:cluster_num]
        centers = x[init_idx].clone()  # (K, D)

        for _ in range(max_iter):
            old_centers = centers.clone()
            dist = pairwise_sq_dist(x, centers)
            labels = dist.argmin(dim=1) 
            new_centers = []
            for k in range(cluster_num):
                mask = (labels == k)
                if mask.any():
                    new_centers.append(x[mask].mean(dim=0))
                else:
                    rand_idx = torch.randint(0, num_samples, (1,), generator=generator, device=compute_device)
                    new_centers.append(x[rand_idx].squeeze(0))
            centers = torch.stack(new_centers, dim=0)  # (K, D)
            center_shift = torch.norm(centers - old_centers, dim=1).max()
            if center_shift <= tol:
                break
        final_dist = pairwise_sq_dist(x, centers)
        min_dist = final_dist.min(dim=1).values
        inertia = min_dist.sum()

        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.clone()
    proto_out = best_centers.transpose(0, 1).contiguous()
    proto_out = proto_out.to(device=original_device, dtype=torch.float32)

    return proto_out

def mixed_sample(proto, sample_ratio=0.1):
    half_sample_ratio = sample_ratio / 2
    avg_proto = average_sample(proto)
    random_proto = random_sample(proto, half_sample_ratio)
    cluster_proto = cluster_sample(proto, half_sample_ratio)
    proto = torch.hstack((avg_proto, random_proto, cluster_proto))
    return proto

def dp(proto, diff_privacy_scale=0, diff_privacy_perturbation=0):
    device = proto.device
    noise = torch.normal(0, diff_privacy_scale, proto.shape)
    noise = noise * diff_privacy_perturbation
    noise = noise.to(device)
    proto = proto + noise
    return proto
def aggregate_and_average_features(client_features, args, sample_ratio=0.1, sample_method='cluster'):
    protos = defaultdict(list)
    
    for client_idx, features in client_features.items():
        for class_idx, feature_list in features.items():
            if len(feature_list) > 1:
                stacked_features = torch.stack(feature_list)
                
                if sample_method == 'average':
                    prototype = torch.mean(stacked_features, dim=0, keepdim=True)
                elif sample_method == 'random':
                    sample_idx = random.sample(range(stacked_features.shape[0]), 
                                             math.ceil(stacked_features.shape[0]*sample_ratio))
                    prototype = stacked_features[sample_idx]
                elif sample_method == 'cluster':
                    prototype = cluster_sample(stacked_features.transpose(0, 1), sample_ratio).transpose(0, 1)
                else:
                    prototype = stacked_features
            
            else:
                prototype = feature_list[0].unsqueeze(0)


            if args.diff_privacy_scale > 0 and args.diff_privacy_perturbation > 0:
                prototype = prototype.transpose(0, 1)
                prototype = dp(prototype,
                               diff_privacy_scale=args.diff_privacy_scale,
                               diff_privacy_perturbation=args.diff_privacy_perturbation)
                prototype = prototype.transpose(0, 1)
            
            protos[class_idx].append(prototype)

    aggregated_protos = {}
    for class_id, features_list in protos.items():
        aggregated_protos[class_id] = torch.cat(features_list, dim=0)

    return aggregated_protos

def generate_protos_training_data(prototype, batchsize=10):
    classes = prototype.keys()
    protos = []
    labels = []


    for c in classes:
        protos_class_c = prototype[c]
        protos.append(protos_class_c)
        labels.extend([c] * protos_class_c.shape[0])

    protos = torch.vstack(protos)
    labels = torch.tensor(labels, dtype=torch.long)
    
    total_protos = protos.shape[0]

    perm = torch.randperm(total_protos)
    protos = protos[perm, :]
    labels = labels[perm]

    max_full_batches = total_protos // batchsize
    new_total_protos = max_full_batches * batchsize

    protos = protos[:new_total_protos]
    labels = labels[:new_total_protos]
    print('protos:', protos.shape)
    print('labels:', labels.shape)
    training_data = []
    for i in range(0, new_total_protos, batchsize):
        training_data.append((protos[i:i+batchsize], labels[i:i+batchsize]))
    total_memory = 0
    for features, labels in training_data:
        features_mem = features.numel() * features.element_size() 
        labels_mem = labels.numel() * labels.element_size()
        total_memory += features_mem + labels_mem

    mem_unit = [(1e9, 'GB'), (1e6, 'MB'), (1e3, 'KB')]
    for divisor, unit in mem_unit:
        if total_memory > divisor:
            print(f"Total Memory Usage: {total_memory/divisor:.2f} {unit}")
            break
    else:
        print(f"Total Memory Usage: {total_memory} Bytes")

    return training_data


class LoraFU(Base):
    def __init__(self, args):
        super(LoraFU, self).__init__(args)
        self.args = args
        self.log_dir = f"logs/moe_{self.args.data_name}_{self.args.alpha}"
        self.param_change_dict = {}
        self.param_size = {}
        

    def extract_clip_features(self, trainloader, device='cuda'):
        model_path = 'save_model/global_model_{}.pth'.format(self.args.data_name)
        model = model_init(self.args)

        if self.args.data_name in ['fashionmnist', 'mnist', 'cifar100', 'cifar10']:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            if self.args.data_name in ['mnist', 'fashionmnist']: 
                model.fc = nn.Identity()
            else:
                if hasattr(model, 'model') and hasattr(model.model, 'fc'):
                    model.model.fc = nn.Identity()
                elif hasattr(model, 'fc'):
                    model.fc = nn.Identity()
                elif hasattr(model, 'linear_head'):
                    model.linear_head = nn.Identity()
            model.eval().to(device)

        elif self.args.data_name == 'text':
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval().to(device)

        else:
            raise ValueError(f"Unsupported data name: {self.args.data_name}")

        protos = defaultdict(list)

        with torch.no_grad():
            for batch in trainloader:
                if self.args.data_name == 'text':
                    if isinstance(batch, dict):
                        input_ids = batch['input_ids'].long().to(device, non_blocking=False)
                        labels = batch['labels'].long().to(device, non_blocking=False)
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None: 
                            attention_mask = attention_mask.long().to(device, non_blocking=False)
                    else:
                        if len(batch) == 3:
                            input_ids, attention_mask, labels = batch
                            input_ids = input_ids.long().to(device, non_blocking=False)
                            attention_mask = attention_mask.long().to(device, non_blocking=False)
                            labels = labels.long().to(device, non_blocking=False)
                        elif len(batch) == 2:
                            input_ids, labels = batch
                            input_ids = input_ids.long().to(device, non_blocking=False)
                            labels = labels.long().to(device, non_blocking=False)
                            attention_mask = (input_ids != 0).long().to(device, non_blocking=False)
                        else:
                            raise ValueError(f"Unexpected batch length: {len(batch)} for text data")

                    if hasattr(model, 'get_features'):
                        if attention_mask is not None:
                            features = model.get_features(input_ids, attention_mask=attention_mask)
                        else:
                            features = model.get_features(input_ids)
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        features = getattr(outputs, 'pooler_output', None)
                        if features is None:
                            hidden_states = getattr(outputs, 'hidden_states', None)
                            if hidden_states is not None and len(hidden_states) > 0:
                                features = hidden_states[-1][:, 0, :]
                            else:
                    
                                features = getattr(outputs, 'logits', outputs)

                    labels_np = labels.cpu().numpy()
                    for i in range(features.size(0)):
                        protos[labels_np[i]].append(features[i].cpu())

                else:
                    images = batch[0].to(device, non_blocking=False)
                    labels = batch[1].to(device, non_blocking=False)
                    features = model(images)
                    features = features.view(features.size(0), -1)
                    labels_np = labels.cpu().numpy()
                    for i in range(features.size(0)):
                        protos[labels_np[i]].append(features[i].cpu())
        # print(f"Extracted feature shape: {features.shape}") 

        return protos


    
        

    def prototype_train_client_lora(self, global_model, client_features, test_loaders):
        
        global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(self.args.data_name)))
        avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, 1, global_model, self.args, test_loaders)
        print('origin-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format('xprototype_train_lorax', avg_r_acc, avg_f_acc))

        if self.args.data_name == 'text': 
            lora_model = Loratext(self.args, global_model)
        else:
            lora_model = Lora(self.args, global_model)

        lora_model = lora_model.to(self.args.device)
        torch.save(lora_model.state_dict(), 'save_model/global_loramodel_{}.pth'.format(self.args.data_name))
        
        checkpoints_ls = []
        result_list = []
        consume_time = 0

        selected_client_features = {client_idx: features for client_idx, features in client_features.items() if client_idx not in self.args.forget_client_idx}

        prototype = aggregate_and_average_features(selected_client_features, self.args, sample_method='cluster', sample_ratio=0.8)
        print("prototype_trained_completed")
        training_data = generate_protos_training_data(prototype)
        print("training_data_trained_completed")
        
        if self.args.data_name in ['fashionmnist', 'mnist']:
            classifier = lora_model.base_model.fc
        elif self.args.data_name in ['cifar10', 'cifar100']:
            base = lora_model.base_model
            if hasattr(base, 'fc'):
                classifier = base.fc
            elif hasattr(base, 'model') and hasattr(base.model, 'fc'):
                classifier = base.model.fc
            elif hasattr(base, 'linear_head'):
                classifier = base.linear_head
        else:
            classifier = lora_model.base_model.model.classifier

        for param in classifier.parameters():
            param.requires_grad = False
            
        criterion_cl = torch.nn.CrossEntropyLoss()
        criterion_re = torch.nn.MSELoss()
        
        optimizer = optim.SGD(lora_model.parameters(), lr=self.args.lr)
        print('\n')
        print(5 * "#" + "  Adapter Federated Client Unlearning Start  " + 5 * "#")
        
        for epoch in range(self.args.lora_trained_epoch):
            start_time = time.time()  

            lora_model.train()
            total_loss = 0
            correct = 0
            total = 0

            for features, labels in training_data: 

                features = features.to(self.args.device)
                labels = labels.to(self.args.device)
                x = features.float() 

                optimizer.zero_grad()
                
                if self.args.data_name in ['fashionmnist', 'mnist']: 
                    adapter = lora_model.adapters['adapter']
                    fc = lora_model.base_model.fc
                elif self.args.data_name in ['cifar10', 'cifar100']:
                    adapter = lora_model.adapters['adapter']
                    if hasattr(lora_model.base_model, 'fc'):
                        fc = lora_model.base_model.fc
                    elif hasattr(lora_model.base_model, 'model') and hasattr(lora_model.base_model.model, 'fc'):
                        fc = lora_model.base_model.model.fc
                    elif hasattr(lora_model.base_model, 'linear_head'):
                        fc = lora_model.base_model.linear_head
                    else:
                        raise AttributeError("Expected classifier head (fc / model.fc / linear_head) for cifar data")
                elif self.args.data_name == 'text': 
                    adapter = lora_model.adapters['adapter']
                    fc = lora_model.classifier
                else:
                    raise NotImplementedError(f"Unsupported data: {self.args.data_name}")
                    
                ada_outputs = adapter(x)
                outputs = fc(ada_outputs)
                
                loss_cl = criterion_cl(outputs, labels)
                loss_re = criterion_re(ada_outputs, x) 
                loss = 0.5*loss_cl + 0.5*loss_re
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            accuracy = 100.* correct / total
            end_time = time.time()  
            epoch_time = end_time - start_time  
            consume_time += epoch_time

            print(f'Epoch [{epoch + 1}/{self.args.lora_trained_epoch}], Loss: {total_loss / len(training_data):.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s, Cumulative: {consume_time:.2f}s')
            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, lora_model, self.args, test_loaders)

            for item in test_result_ls:
                item.append(consume_time)

            print('Adapter_unlearning_model_Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_r_acc, avg_f_acc))
            result_list.extend(test_result_ls)
        
        df = pd.DataFrame(
            result_list,
            columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss', 'Comsume_time']
        )

        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv('./results/{}/Acc_loss_adapter_{}_data_{}_distri_{}_fnum_{}.csv'.format(
                    self.args.forget_paradigm,
                    self.args.forget_paradigm,
                    self.args.data_name,
                    self.args.alpha,
                    len(self.args.forget_client_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_adapter_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(
                        self.args.forget_paradigm,
                        self.args.forget_paradigm,
                        self.args.data_name,
                        self.args.alpha,
                        len(self.args.forget_client_idx), self.args.cut_sample))

        print(5 * "#" + "  Adapter Federated Client Unlearning End  " + 5 * "#")
        return lora_model



    def prototype_train_class_lora(self, global_model, reversed_classes_features, test_loaders):
        global_model.load_state_dict(
            torch.load('save_model/global_model_{}.pth'.format(self.args.data_name))
        )

        if self.args.data_name == 'text':
            lora_model = Loratext(self.args, global_model)
        else:
            lora_model = Lora(self.args, global_model)

        lora_model = lora_model.to(self.args.device)
        torch.save(
            lora_model.state_dict(),
            'save_model/global_loramodel_{}.pth'.format(self.args.data_name)
        )

        checkpoints_ls = []
        result_list = []
        consume_time = 0.0 

        prototype = aggregate_and_average_features(
            reversed_classes_features,
            self.args,
            sample_method='cluster',
            sample_ratio=0.1
        )
        print("prototype_trained_completed")

        training_data = generate_protos_training_data(prototype)
        print("training_data_trained_completed")

        if self.args.data_name in ['fashionmnist', 'mnist']:
            classifier = lora_model.base_model.fc
        elif self.args.data_name in ['cifar10', 'cifar100']:
            base = lora_model.base_model
            if hasattr(base, 'fc'):
                classifier = base.fc
            elif hasattr(base, 'model') and hasattr(base.model, 'fc'):
                classifier = base.model.fc
            elif hasattr(base, 'linear_head'):
                classifier = base.linear_head
            else:
                raise AttributeError("Expected classifier head (fc / model.fc / linear_head) for cifar data")
        else:
            classifier = lora_model.base_model.model.classifier

        for param in classifier.parameters():
            param.requires_grad = False

        criterion_cl = torch.nn.CrossEntropyLoss()
        criterion_re = torch.nn.MSELoss()
        optimizer = optim.SGD(lora_model.parameters(), lr=self.args.lr)

        print('\n')
        print(5 * "#" + "  Adapter Federated Client Unlearning Start  " + 5 * "#")

        for epoch in range(self.args.lora_trained_epoch):
            start_time = time.time()

            lora_model.train()
            total_loss = 0
            correct = 0
            total = 0

            for features, labels in training_data:
                features = features.to(self.args.device)
                labels = labels.to(self.args.device)
                x = features.float()

                optimizer.zero_grad()

                if self.args.model == 'LeNet_FashionMNIST':
                    adapter = lora_model.adapters['adapter']
                    fc = lora_model.base_model.fc
                    ada_outputs = adapter(x)
                    outputs = fc(ada_outputs)

                elif self.args.data_name in ['cifar10', 'cifar100']:
                    adapter = lora_model.adapters['adapter']
                    base = lora_model.base_model
                    if hasattr(base, 'fc'):
                        fc = base.fc
                    elif hasattr(base, 'model') and hasattr(base.model, 'fc'):
                        fc = base.model.fc
                    elif hasattr(base, 'linear_head'):
                        fc = base.linear_head
                    else:
                        raise AttributeError("Expected classifier head (fc / model.fc / linear_head) for cifar data")
                    ada_outputs = adapter(x)
                    outputs = fc(ada_outputs)
                
                elif self.args.data_name == 'text':
                    adapter = lora_model.adapters['adapter']
                    classifier = lora_model.classifier 
                    ada_outputs = adapter(x)        
                    outputs = classifier(ada_outputs) 

                loss_cl = criterion_cl(outputs, labels)
                loss_re = criterion_re(ada_outputs, x)
                loss = 0.5*loss_cl + 0.5*loss_re

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            accuracy = 100.* correct / total

            end_time = time.time()
            epoch_time = end_time - start_time
            consume_time += epoch_time 

            print(
                f'Adapter_Epoch [{epoch}/{self.args.lora_trained_epoch}], '
                f'Loss: {total_loss / len(training_data):.4f}, '
                f'Accuracy: {accuracy:.2f}%, '
                f'EpochTime: {epoch_time:.2f}s, '
                f'ConsumeTime: {consume_time:.2f}s'
            )

            avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(
                self, epoch, lora_model, self.args, test_loaders
            )

            print(
                'Adapter_Epoch={}, Remember Test Acc={}, Forget Test Acc={}'
                .format(epoch, avg_r_acc, avg_f_acc)
            )

            for item in test_result_ls:

                item.append(consume_time)

            result_list.extend(test_result_ls)


        df = pd.DataFrame(
            result_list,
            columns=[
                'Epoch',
                'Client_id',
                'Class_id',
                'Test_acc',
                'Test_loss',
                'Consume_time'
            ]
        )


        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/adapter_64_Acc_loss_adapter_{}_data_{}_distri_{}_fnum_{}.csv'
                    .format(
                        self.args.forget_paradigm,
                        self.args.forget_paradigm,
                        self.args.data_name,
                        self.args.alpha,
                        len(self.args.forget_class_idx)
                    ),
                    index=False
                )
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/adapter_64_Acc_loss_adapter_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'
                    .format(
                        self.args.forget_paradigm,
                        self.args.forget_paradigm,
                        self.args.data_name,
                        self.args.alpha,
                        len(self.args.forget_class_idx),
                        self.args.cut_sample
                    ),
                    index=False
                )

        print(5 * "#" + "  Adapter Federated Class Unlearning End  " + 5 * "#")
        return lora_model


    def prototype_train_sample_lora(self, global_model, reversed_classes_features, test_loaders):
        global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(self.args.data_name)))

        if self.args.data_name == 'text': 
            lora_model = Loratext(self.args, global_model)
        else:
            lora_model = Lora(self.args, global_model)

        lora_model = lora_model.to(self.args.device)
        torch.save(lora_model.state_dict(), 'save_model/global_loramodel_{}.pth'.format(self.args.data_name))
        
        checkpoints_ls = []
        result_list = []
        consume_time = 0
        
        prototype = aggregate_and_average_features(reversed_classes_features, self.args, sample_method='cluster', sample_ratio=0.5)
        print("prototype_trained_completed")
        training_data = generate_protos_training_data(prototype)
        print("training_data_trained_completed")
        if self.args.data_name in ['fashionmnist', 'mnist']:
            classifier = lora_model.base_model.fc
        elif self.args.data_name in ['cifar10', 'cifar100']:
            base = lora_model.base_model
            if hasattr(base, 'fc'):
                classifier = base.fc
            elif hasattr(base, 'model') and hasattr(base.model, 'fc'):
                classifier = base.model.fc
            elif hasattr(base, 'linear_head'):
                classifier = base.linear_head
            else:
                raise AttributeError("Expected classifier head (fc / model.fc / linear_head) for cifar data")
        else:
            classifier = lora_model.base_model.model.classifier

        for param in classifier.parameters():
            param.requires_grad = False
            
        criterion_cl = torch.nn.CrossEntropyLoss()
        criterion_re = torch.nn.MSELoss()
        optimizer = optim.SGD(lora_model.parameters(), lr=self.args.lr)
        print('\n')
        print(5 * "#" + "  Adapter Federated Client Unlearning Start  " + 5 * "#")

        for epoch in range(self.args.lora_trained_epoch):
            start_time = time.time()  

            lora_model.train()
            total_loss = 0
            correct = 0
            total = 0

            for features, labels in training_data: 
                features = features.to(self.args.device)
                labels = labels.to(self.args.device)
                
                x = features.float() 

                optimizer.zero_grad()
                if self.args.data_name in ['fashionmnist', 'mnist']:
                    adapter = lora_model.adapters['adapter']
                    fc = lora_model.base_model.fc
                    ada_outputs = adapter(x)
                    outputs = fc(ada_outputs)
                elif self.args.data_name in ['cifar10', 'cifar100']:
                    adapter = lora_model.adapters['adapter']
                    base = lora_model.base_model
                    if hasattr(base, 'fc'):
                        fc = base.fc
                    elif hasattr(base, 'model') and hasattr(base.model, 'fc'):
                        fc = base.model.fc
                    elif hasattr(base, 'linear_head'):
                        fc = base.linear_head
                    ada_outputs = adapter(x)
                    outputs = fc(ada_outputs)
                elif self.args.data_name == 'text':
                    adapter = lora_model.adapters['adapter']
                    fc = lora_model.classifier
                    ada_outputs = adapter(x)
                    outputs = fc(ada_outputs)
                else:
                    raise NotImplementedError(f"Unsupported data: {self.args.data_name}")
                
                loss_cl = criterion_cl(outputs, labels)
                loss_re = criterion_re(ada_outputs, x) 
                loss = 0.5*loss_cl + 0.5*loss_re
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            end_time = time.time()  
            epoch_time = end_time - start_time  
            accuracy = 100.* correct / total
            consume_time += epoch_time
            print(f'Adapter_Epoch [{epoch}/{self.args.lora_trained_epoch}], Loss: {total_loss / len(training_data):.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s, Cumulative: {consume_time:.2f}s')
            avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, lora_model, self.args, test_loaders)

            for item in test_result_ls:
                item.append(consume_time)

            result_list.extend(test_result_ls)
            print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc))

        df = pd.DataFrame(
            result_list,
            columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc', 'Comsume_time']
        )
        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_adapter_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_adapter_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                        self.args.forget_paradigm,
                                                                                        self.args.data_name,
                                                                                        self.args.alpha,
                                                                                        len(self.args.forget_class_idx), self.args.cut_sample))

        print(5 * "#" + "  Lora Federated Sample Unlearning End  " + 5 * "#")
        return lora_model


    def train_normal(self, global_model, client_all_loaders, test_loaders):
        print('\n')
        print(5 * "#" + "  LoraFU Federated Training Start  " + 5 * "#")
        deltas = defaultdict(dict)
        checkpoints_ls = []
        result_list = []
        param_list = []
        start_time = time.time()
        for epoch in range(self.args.global_epoch):
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user*self.args.fraction), replace=False))
            select_client_loaders = [client_all_loaders[idx] for idx in selected_clients]
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            global_model = self.fedavg(client_models)
            if epoch == 0:
                client_features = {client_idx: defaultdict(list) for client_idx in range(self.args.num_user)}
                print(client_features)
        
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            global_model = self.fedavg(client_models)
            if self.args.global_epoch - epoch == 1:
                torch.save(global_model.state_dict(), 'save_model/global_model_{}.pth'.format(self.args.data_name))
            
            if self.args.global_epoch - epoch <= 1:
                for client_idx in selected_clients:
                    client_loader = client_all_loaders[client_idx]
                    
                    client_protos = self.extract_clip_features(client_loader, device=self.args.device)
                    
                    for class_idx, features in client_protos.items():
                        if class_idx not in client_features[client_idx]:
                            client_features[client_idx][class_idx] = features
                        else:
                            client_features[client_idx][class_idx] = torch.cat([
                                client_features[client_idx][class_idx], 
                                features
                            ], dim=0)
            
            all_idx = list(range(self.args.num_user))

            client_test_acc = []
        
            if self.args.forget_paradigm == 'sample': 
            
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model,
                                                                                            self.args, test_loaders)
                print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero,
                                                                                avg_test_acc))
                result_list.extend(test_result_ls)
            
            elif self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders)
                print('origin_model_Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)
            
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)

        end_time = time.time()
        print('time cost: {}'.format(end_time - start_time))

        if self.args.forget_paradigm == 'sample': 
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        elif self.args.forget_paradigm == 'client':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        elif self.args.forget_paradigm == 'class':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        if self.args.save_normal_result:
            df.to_csv('./results/Acc_loss_fl_{}_data_{}_distri_{}.csv'.format(self.args.forget_paradigm, self.args.data_name, self.args.alpha))

        return client_features, global_model, client_models

    
    def forget_client_train(self, global_model, client_all_loaders, test_loaders):
        global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(self.args.data_name)))
        avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, 1, global_model, self.args,
                                                                  test_loaders)
        print('Lora-model-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format('unlearning_model', avg_r_acc,
                                                                                 avg_f_acc))
        if self.args.data_name == 'text':
            lora_model = Loratext(self.args, global_model)
        else:
            lora_model = Lora(self.args, global_model)
        #torch.save(lora_model.state_dict(), 'save_model/global_loramodel_{}.pth'.format(self.args.data_name))
        print('\n')
        print(5 * "#" + "  shade model forget class Federated training shade model Start  " + 5 * "#")
        checkpoints_ls = []
        result_list = []
        consume_time = 0

        for epoch in range(self.args.global_epoch):
            lora_model.train()
            selected_clients = [i for i in range(self.args.num_user) if i not in self.args.forget_client_idx]
            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)

            std_time = time.time()
            client_models = self.global_train_once(epoch, lora_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            avg_model = self.fedavg(client_models)
            consume_time += end_time - std_time
            lora_model.load_state_dict(avg_model.state_dict())

            lora_model.eval()


            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, lora_model, self.args,
                                                                      test_loaders)

            result_list.extend(test_result_ls)

            print('Lora-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format(epoch, avg_r_acc,
                                                                                    avg_f_acc))
        print(5 * "#" + "  shade model forget class Federated training shade model end  " + 5 * "#")                                                                       
        return lora_model
    
    


    def forget_class(self, global_model, client_all_loaders, test_loaders):
        print('\n')
        print(5 * "#" + "  shade model forget class Federated Class Unlearning Start  " + 5 * "#")
        num_selected_clients = self.args.num_user * self.args.forget_client_idx
        checkpoints_ls = []
        result_list = []
        consume_time = 0
        if self.args.data_name == 'text':
            lora_model = Loratext(self.args, global_model)
        else:
            lora_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            lora_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))

            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)
            std_time = time.time()

            client_models = self.global_train_once(epoch, lora_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            lora_model = self.fedavg(client_models)
            consume_time += end_time-std_time
            avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, lora_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, Remember Test Acc={}, Forget Test Acc={}'.format(epoch, avg_r_acc, avg_f_acc))
        print(5 * "#" + "  shade model forget class Federated Class Unlearning End  " + 5 * "#")
        return lora_model

    def forget_sample(self, global_model, client_all_loaders, test_loaders):
        print('\n')
        print(5 * "#" + "  shade model forget sample Federated Sample Unlearning Start  " + 5 * "#")
        checkpoints_ls = []
        result_list = []
        consume_time = 0
        if self.args.data_name == 'text':
            lora_model = Loratext(self.args, global_model)
        else:
            lora_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            lora_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))# 将需要遗忘的客户端排除在外

            self.select_forget_idx = list()
            select_client_loaders = list()
            record = -1
            for idx in selected_clients:
                select_client_loaders.append(client_all_loaders[idx])
                record += 1
                if idx in self.args.forget_client_idx:
                    self.select_forget_idx.append(record)
            std_time = time.time()
            client_models = self.global_train_once(epoch, lora_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            lora_model = self.fedavg(client_models)
            consume_time += end_time-std_time
            avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, lora_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc))
        print(5 * "#" + " shade model forget sample Federated Sample Unlearning End  " + 5 * "#")
        return lora_model


    def distill(self, proxy_data, teacher_model, student_model, test_loaders):
        student_model.to(self.args.device)
        optimizer = optim.SGD(student_model.gate_model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        all_idx = [idx for idx in range(self.args.num_user)]

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        avg_acc = 0
        result_list = []
        std_time = time.time()
        for server_epoch in range(self.args.distill_epoch):
            last_avg_acc = avg_acc
            for batch_idx, (data, target) in enumerate(proxy_data):
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                loss_all = []
                loss_all_f = []
                loss_all_r = []
                z_r_ls = []
                z_f_ls = []
                weights_f = []
                weights_r = []
                student_outputs, j = student_model(data)
                hard_loss = nn.CrossEntropyLoss()(student_outputs, target)
                for k, teacher in enumerate(teacher_model):
                    teacher.to(self.args.device)
                    teacher_outputs, i = teacher(data)

                   
                    if self.args.if_unlearning == True:
                        if k not in self.args.forget_client_idx:
                            distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                                nn.functional.log_softmax(student_outputs / self.args.temperature, dim=1),
                                nn.functional.softmax(teacher_outputs / self.args.temperature, dim=1))
                            # distillation_loss = nn.CrossEntropyLoss(student_outputs, teacher_outputs)
                            loss_all.append(hard_loss * 0.5 + distillation_loss * 0.5)
                        else:
                            distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                            nn.functional.log_softmax(student_outputs / self.args.temperature, dim=1),
                            nn.functional.softmax(teacher_outputs / self.args.temperature, dim=1))
                        loss_all.append(hard_loss * 0.3 + distillation_loss * 0.7)

                
                optimizer.zero_grad()
                loss_avg = sum(loss_all)/len(loss_all)
                loss_avg.backward()
                optimizer.step()
            scheduler.step()
            end_time = time.time()
            consume_time = end_time-std_time

            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, server_epoch, student_model, self.args,
                                                                      test_loaders)
            result_list.extend(test_result_ls)
            if self.args.if_unlearning == True:
                print('MoE-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format(server_epoch, avg_r_acc, avg_f_acc))
            else:
                avg_acc = np.array([row[2] for row in test_result_ls])
                avg_acc = np.mean(avg_acc)
                print('Distill Acc: {}'.format(avg_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        df.to_csv(
                './results/Acc_loss_normalmoe_after_distill_{}_distri_{}.csv'.format(self.args.data_name, self.args.alpha))

        return student_model, avg_acc, df


    def relearn_unlearning_knowledge(self, unlearning_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        all_global_models = list()
        all_client_models = list()
        global_model = unlearning_model
        result_list = []

        all_global_models.append(global_model)
        std_time = time.time()
        for epoch in range(self.args.global_epoch):
            if self.args.forget_paradigm == 'client':
                select_client_loaders = list()
                for idx in self.args.forget_client_idx:
                    select_client_loaders.append(client_all_loaders[idx])
            elif self.args.forget_paradigm == 'class':
                select_client_loaders = list()
                client_loaders = select_forget_class(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            elif self.args.forget_paradigm == 'sample':
                select_client_loaders = list()
                client_loaders = select_forget_sample(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args,
                                                   checkpoints_ls)

            all_client_models += client_models
            global_model = self.fedavg(client_models)
            all_global_models.append(copy.deepcopy(global_model).to('cpu'))
            end_time = time.time()

            consume_time = end_time - std_time

            if self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args,
                                                                          test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss',
                                           'Comsume_time'])
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args,
                                                                         test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss', 'Comsume_time'])
            elif self.args.forget_paradigm == 'sample':
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model, self.args, test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc', 'Comsume_time'])

            global_model.to('cpu')

            print("Relearn Round = {}".format(epoch))
        
        if self.args.cut_sample == 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm,
                                                                                      ), index=False)
        elif self.args.cut_sample < 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm,
                                                                                       self.args.cut_sample), index=False)
        return