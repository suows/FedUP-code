import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from dataset.data_utils import data_set, separate_data, split_proxy
import numpy as np
import os
import pickle as pkl
import pandas as pd
import tqdm
import random
from transformers import BertTokenizer
from transformers import AutoTokenizer

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'


def data_init(FL_params):
    kwargs = {'num_workers': 8, 'pin_memory': True} if FL_params.device == 'cuda' else {'num_workers': 8}
    dataset_x = []
    dataset_at = []
    dataset_y = []
    if FL_params.data_name == 'text':
        trainset, testset = data_set_text(FL_params, True)
        test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)
        for sample in trainset:
            dataset_x.append(sample['input_ids'].numpy())
            dataset_at.append(sample['attention_mask'].numpy())
            dataset_y.append(sample['label'].item()) 
        if FL_params.forget_paradigm == 'client':
            for sample in testset:
                dataset_x.append(sample['input_ids'].numpy())
                dataset_at.append(sample['attention_mask'].numpy())
                dataset_y.append(sample['label'].item())

    else:
        trainset, testset = data_set(FL_params.data_name)
        test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)
        train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)

        for train_data in train_loader:
            x_train, y_train = train_data
            dataset_x.extend(x_train.cpu().detach().numpy())
            dataset_y.extend(y_train.cpu().detach().numpy())
        if FL_params.forget_paradigm == 'client':
            for test_data in test_loader:
                x_test, y_test = test_data
                dataset_x.extend(x_test.cpu().detach().numpy())
                dataset_y.extend(y_test.cpu().detach().numpy())

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)
    if FL_params.data_name == 'text':
        dataset_at = np.array(dataset_at)

        X, AT, y, statistic = separate_data((dataset_x, dataset_at, dataset_y), FL_params.num_user, FL_params.num_classes, FL_params,
                                        FL_params.niid, FL_params.balance, FL_params.partition, class_per_client=2)
        client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = split_proxy(X, y, FL_params, AT)
    else:
        X, y, statistic = separate_data((dataset_x, dataset_y), FL_params.num_user, FL_params.num_classes, FL_params,
                                        FL_params.niid, FL_params.balance, FL_params.partition, class_per_client=2)

        client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = split_proxy(X, y, FL_params)
    FL_params.datasize_ls = [len(k) for k in X]
    if FL_params.forget_paradigm == 'client':
        test_loaders = test_loaders
        proxy_test_loaders = proxy_test_loaders
    else:
        proxy_test_x = []
        proxy_test_at = []
        proxy_test_y = []
        
        for batch in test_loader:
            if FL_params.data_name == 'text':
                proxy_test_x.append(batch['input_ids'])
                proxy_test_at.append(batch['attention_mask'])
                proxy_test_y.append(batch['label'])
            else:
                x, y = batch
                proxy_test_x.append(x)
                proxy_test_y.append(y)

        proxy_test_x = torch.cat(proxy_test_x)
        proxy_test_y = torch.cat(proxy_test_y)
        
        if FL_params.data_name == 'text':
            proxy_test_at = torch.cat(proxy_test_at)
            proxy_test_dataset = TensorDataset(proxy_test_x, proxy_test_at, proxy_test_y)
        else:
            proxy_test_dataset = TensorDataset(proxy_test_x, proxy_test_y)
            
        proxy_test_loader = DataLoader(
            proxy_test_dataset,
            batch_size=FL_params.test_batch_size,
            shuffle=True,
            **kwargs
        )
        proxy_test_loaders = [proxy_test_loader for _ in range(FL_params.num_user)]
    print(">>> Data per client:", [len(x) for x in X])
    return client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders

def cross_data_init(FL_params):
    kwargs = {'num_workers': 8} if FL_params.device == 'cuda' else {'num_workers': 8}
    dataset_x = []
    dataset_y = []
    if FL_params.data_name == 'text':
        _, trainset, testset = data_set_text(FL_params, True)
        for (x, y) in trainset:
            dataset_x.append(x)
            dataset_y.append(y)
        for (x, y) in testset:
            dataset_x.append(x)
            dataset_y.append(y)
    else:
        trainset, testset = data_set(FL_params.data_name)
        test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=8, **kwargs)
        train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=8,
                                  **kwargs)

        for train_data in train_loader:
            x_train, y_train = train_data
            dataset_x.extend(x_train.cpu().detach().numpy())
            dataset_y.extend(y_train.cpu().detach().numpy())
        if FL_params.forget_paradigm == 'client':
            for test_data in test_loader:
                x_test, y_test = test_data
                dataset_x.extend(x_test.cpu().detach().numpy())
                dataset_y.extend(y_test.cpu().detach().numpy())

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    class_num = int(FL_params.num_classes/FL_params.num_user)
    X = []
    y = []
    idx_ls = []
    for user in range(FL_params.num_user):
        idx = []
        for i in range(class_num):
            item = user*class_num + i
            indices = [idx for idx, label in enumerate(dataset_y) if label == item]
            idx.extend(indices)
        idx_ls.append(idx)
    corss_idx = idx_ls[0][:int(len(idx_ls[0])*0.01)]
    idx_ls[0] = idx_ls[0][int(len(idx_ls[0])*0.01):]
    idx_ls[1] = corss_idx + idx_ls[1]
    remain_idx = []
    for idx in range(1, FL_params.num_user):
        remain_idx.extend(idx_ls[idx])
    random.shuffle(remain_idx)
    sublist_size = len(remain_idx) // (FL_params.num_user-len(FL_params.forget_client_idx))
    remainder = len(remain_idx) % (FL_params.num_user-len(FL_params.forget_client_idx))
    sublists = [remain_idx[i * sublist_size + min(i, remainder):(i + 1) * sublist_size + min(i + 1, remainder)] for i in
                range(9)]

    for idx in range(1, FL_params.num_user):
        idx_ls[idx] = sublists[idx-1]

    for user in range(FL_params.num_user):
        X.append(dataset_x[idx_ls[user]])
        y.append(dataset_y[idx_ls[user]])

    for i in range(FL_params.num_user):
        print('client {} data size {} lable {}'.format(i, len(X[i]),np.unique(y[i])))

    client_loaders, test_loaders, proxy_loader = split_proxy(X, y, FL_params)
    FL_params.datasize_ls = [len(k) for k in X]
    if FL_params.forget_paradigm == 'client':
        test_loaders = test_loaders
    else:
        test_loaders = [test_loader for _ in range(FL_params.num_user)]

    return client_loaders, test_loaders, proxy_loader

def data_set_text(config, unused_flag=True):
    local_model_path = "/root/autodl-tmp/FedUP_code/FedUP/models/bert-tiny-local"
    print(">>> USING NEW HF AG_NEWS LOADER <<<")
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer
    import os
    if not hasattr(config, 'pad_size'):
        config.pad_size = 128
    cache_dir = './dataset/text'
    os.makedirs(cache_dir, exist_ok=True)
    processed_train_path = os.path.join(cache_dir, 'tokenized_train')
    processed_test_path = os.path.join(cache_dir, 'tokenized_test')
    if os. path.exists(processed_train_path) and os.path.exists(processed_test_path):
        print("Loading preprocessed dataset from disk...")
        tokenized_train = load_from_disk(processed_train_path)
        tokenized_test = load_from_disk(processed_test_path)
    else:
        print("Preprocessed data not found, downloading and processing...")
        tokenizer = AutoTokenizer. from_pretrained(local_model_path)
        print("Loading AG News dataset...")
        dataset = load_dataset('ag_news', cache_dir=cache_dir)

        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=config.pad_size,
                return_tensors='pt'
            )
            
        print("Tokenizing train set...")
        tokenized_train = dataset['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        print("Tokenizing test set...")
        tokenized_test = dataset['test'].map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        print("Saving preprocessed dataset to disk...")
        tokenized_train.save_to_disk(processed_train_path)
        tokenized_test.save_to_disk(processed_test_path)
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    print(f"Train set size: {len(tokenized_train)}")
    print(f"Test set size: {len(tokenized_test)}")
    return tokenized_train, tokenized_test

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt' 
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),     
            'attention_mask': encoding['attention_mask'].squeeze(0), 
            'labels': torch.tensor(label, dtype=torch.long) 
        }
