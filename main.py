
import argparse
import copy
from collections import defaultdict
from dataset.generate_data import data_init, cross_data_init
import torch
import cProfile
from algs import my_forget, fl_base
from utils import *
import random
import numpy as np
from models.Model_base import Lora

def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    # TODO
    parser.add_argument('--model', type=str, required=False, default='ViT_Cifar10', help= 'choose a model: LeNet_FashionMNIST,CNN_Cifar10,CNN_Cifar100')
    parser.add_argument('--data_name', type=str, required=False, default='cifar10', help= 'choose: mnist, fashionmnist, purchase, adult, cifar10, text, cifar100, cifar10')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('--distribution', default=True, type=bool, help='True means iid, while False means non-iid')
    parser.add_argument('--train_with_test', default=True, type=bool, help='')
    parser.add_argument('--temperature', default=0.5, type=float, help='the temperature for distillation loss')
    parser.add_argument('--max_checkpoints', default=3, type=int)

    # TODO
    parser.add_argument('--forget_paradigm', default='client', type=str, help='choose from client or class')
    parser.add_argument('--paradigm', default='adapter', type=str,
                        help='choose the training paradigm:lora, federaser, retrain, infocom22, exactfun, fl, eraseclient, fedau, adapter')
    parser.add_argument('--forget_client_idx', type=list, default=[0]) 
    parser.add_argument('--forget_class_idx', type=int, nargs='+', default=[],
                    help='List of class indices to forget, e.g., --forget_class_idx 0 1 2')
    parser.add_argument('--if_retrain', default=False, type=bool, help='')
    parser.add_argument('--if_unlearning', default=False, type=bool, help='')
    parser.add_argument('--baizhanting', default=True, type=bool, help='')
    parser.add_argument('--backdoor', default=False, type=bool, help='')
    parser.add_argument('--backdoor_frac', default=0.2, type=float, help='')

    # TODO
    parser.add_argument('--MIT', default=True, type=bool, help='whether to use membership inference attack')
    parser.add_argument('--n_shadow', default=1, type=int, help='the number of shadow model')
    parser.add_argument('--cut_sample', default=1.0, type=float, help='using part of the training data')
    parser.add_argument('--relearn', default=True, type=bool, help='whether to relearn the unlearned knowledge')
    parser.add_argument('--save_normal_result', default=True, type=bool, help='whether to save the normal result')
    
    parser.add_argument('--local_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    

    # TODO
    parser.add_argument('--global_epoch', default=1, type=int)
    parser.add_argument('--lora_trained_epoch', default=50, type=int)
    parser.add_argument('--local_epoch', default=1, type=int)
    parser.add_argument('--distill_epoch', default=10, type=int)
    parser.add_argument('--distill_pretrain_epoch', default=2, type=int)
    parser.add_argument('--fraction', default=1.0, type=float, help='the fraction of training data')
    parser.add_argument('--num_user', default=1, type=int)
    
    parser.add_argument('--niid', default=True, type=bool, help='')
    parser.add_argument('--balance', default=True, type=bool, help='')
    parser.add_argument('--partition', default='dir', type=str, help='choose from pat or dir')
    parser.add_argument('--alpha', default=1.0, type=float, help='for Dirichlet distribution')
    parser.add_argument('--proxy_frac', default=0.2, type=float, help='the fraction of training data')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--unlearn_interval', default=1, type=int, help='')
    parser.add_argument('--forget_local_epoch_ratio', default=0.2, type=float)


    parser.add_argument('--epoch_unlearn', default=50, type=int, help='')
    parser.add_argument('--num_iterations', default=50, type=int, help='')
    parser.add_argument('--out_dim', type=int, default=32, help='Adapter Output Dimension')

    parser.add_argument('--dp', action='store_true', default=False, help='whether dp')
    parser.add_argument('--sigma',  type=float, default= 0.1 , help='the sgd of Gaussian noise')
    parser.add_argument('--ul_client_gamma', type=float, default=0.5, help='ul_client_gamma')
    parser.add_argument('--ul_samples_alpha', type=float, default=0.9, help='ul_samples_alpha')
    parser.add_argument('--diff_privacy_scale', type=float, default=0, help='ul_samples_alpha')
    parser.add_argument('--diff_privacy_perturbation', type=float, default=0, help='ul_samples_alpha')

    args = parser.parse_args()
    return args



def set_random_seed(seed=42):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = get_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', args.device)
    model = model_init(args)

    client_all_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = data_init(args)
    print(test_loaders[0])

    args.if_unlearning = False
    case = my_forget.LoraFU(args)
    client_features = {}
    if args.forget_paradigm == 'client':
        client_all_loaders_process, test_loaders_process = baizhanting_attack(args, copy.deepcopy(client_all_loaders),
                                                                                copy.deepcopy(test_loaders))
        proxy_client_loaders_process, proxy_test_loaders_process = baizhanting_attack(args, copy.deepcopy(
            proxy_client_loaders), copy.deepcopy(proxy_test_loaders))
        client_features, model, all_client_models  = case.train_normal(model, client_all_loaders_process, test_loaders_process)

        args.if_unlearning = True
        
        unlearning_model = case.prototype_train_client_lora(copy.deepcopy(model), client_features,
                                                    test_loaders_process)
        if args.MIT:
            args.save_normal_result = False
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_process,
                                        test_loaders, proxy_client_loaders_process, proxy_client_loaders,
                                        proxy_test_loaders_process)
            args.save_normal_result = True
        if args.relearn:
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_process, test_loaders_process)
    elif args.forget_paradigm == 'class':
        client_all_loaders_bk = copy.deepcopy(client_all_loaders)
        proxy_client_loaders_bk = copy.deepcopy(proxy_client_loaders)
        client_features, model, all_client_models = case.train_normal(model, copy.deepcopy(client_all_loaders), test_loaders)
        args.if_unlearning = True 

        reversed_classes_features = defaultdict(lambda: defaultdict(list))
        for client_idx, features in client_features.items():
            for class_id, feature_list in features.items():
                if class_id in args.forget_class_idx:
                    label_ls = [i for i in range(args.num_classes)]
                    label_ls.remove(class_id)
                    inverse_label = np.random.choice(label_ls)
                    reversed_classes_features[client_idx][inverse_label].extend(feature_list)
                else:
                    reversed_classes_features[client_idx][class_id].extend(feature_list)
        for client_idx, features in reversed_classes_features.items():
            for class_id, features_list in features.items():
                print(f'client {client_idx}, label {class_id} num features: {len(features_list)}')
        proxy_train_ls = []
        for user in range(args.num_user):
            for batch in proxy_client_loaders[user]:
                if isinstance(batch, dict):
                    data = batch['input_ids']
                    targets = batch.get('labels') or batch.get('label')
                elif isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        data, targets = batch
                    elif len(batch) >= 3:
                        data = batch[0]
                        targets = batch[2] 
                    else:
                        raise ValueError(f"Unexpected batch length: {len(batch)}")
                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}")

                targets = torch.as_tensor(targets).view(-1)

                for idx, label in enumerate(targets):
                    label_val = label.item()
                    if label_val in args.forget_class_idx:
                        label_ls = [i for i in range(args.num_classes)]
                        label_ls.remove(label_val)
                        inverse_label = np.random.choice(label_ls)
                        label_val = inverse_label
                    d = data[idx]
                    if not isinstance(d, torch.Tensor):
                        d = torch.tensor(d)
                    else:
                        d = d.clone().detach()

                    proxy_train_ls.append((d, torch.tensor(label_val).long()))

        proxy_train_loader = DataLoader(proxy_train_ls, batch_size=args.test_batch_size, shuffle=True)
        unlearning_model = case.prototype_train_class_lora(copy.deepcopy(model), reversed_classes_features, test_loaders)

        if args.MIT:
            args.save_normal_result = False
            membership_inference_attack(
                args, unlearning_model, case, copy.deepcopy(model),
                copy.deepcopy(client_all_loaders_bk), test_loaders,
                proxy_client_loaders_bk, proxy_client_loaders, proxy_test_loaders
            )
            args.save_normal_result = True

        if args.relearn:
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_bk, test_loaders)

    elif args.forget_paradigm == 'sample':
        client_all_loaders_attack = backdoor_attack(args, copy.deepcopy(client_all_loaders))
        proxy_client_loaders_attack = backdoor_attack(args, copy.deepcopy(proxy_client_loaders))
        origin_client_features, model, all_client_models  = case.train_normal(model, client_all_loaders_attack, test_loaders)
        args.if_unlearning = True
        client_all_loaders_process = erase_backdoor(args, copy.deepcopy(client_all_loaders))
        client_features = defaultdict(lambda: defaultdict(list))
        for client_idx in range(args.num_user):
            client_loader = client_all_loaders_process[client_idx]
            client_protos = case.extract_clip_features(client_loader, device=args.device)
            for class_idx, features in client_protos.items():
                client_features[client_idx][class_idx].extend(features)

        print("============================\n")
        proxy_client_loaders_process = erase_backdoor(args, copy.deepcopy(proxy_client_loaders))
        unlearning_model = case.prototype_train_sample_lora(copy.deepcopy(model), client_features, test_loaders)
        if args.MIT:
            args.save_normal_result = False
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_attack,
                                        test_loaders, proxy_client_loaders_attack, proxy_client_loaders_process,
                                        proxy_test_loaders)
            args.save_normal_result = True
        if args.relearn:
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_attack, test_loaders)
