import os
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
from pathlib import Path
import yaml

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_dataset(data_name, path='./data'):
    if not data_name in ['mnist', 'cifar10']:
        raise TypeError('data_name should be a string, including mnist,cifar10. ')

    # model: 2 conv. layers followed by 2 FC layers
    if (data_name == 'mnist'):
        trainset = datasets.MNIST(path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
        testset = datasets.MNIST(path, train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))

    # model: ResNet-50
    elif (data_name == 'cifar10'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = datasets.CIFAR10(root=path, train=True,
                                    download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=path, train=False,
                                   download=True, transform=transform_test)
    return trainset, testset


def get_dataloader(trainset, testset, batch_size, device):
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def split_class_data(dataset, forget_class, num_forget):
    forget_index = []
    class_remain_index = []
    remain_index = []
    sum = 0
    for i, (data, target) in enumerate(dataset):
        if target == forget_class and sum < num_forget:
            forget_index.append(i)
            sum += 1
        elif target == forget_class and sum >= num_forget:
            class_remain_index.append(i)
            remain_index.append(i)
            sum += 1
        else:
            remain_index.append(i)
    return forget_index, remain_index, class_remain_index


# def split_dataset(dataset, forget_class):


def get_unlearn_loader(trainset, testset, forget_class, batch_size, num_forget, repair_num_ratio=0.01):
    train_forget_index, train_remain_index, class_remain_index = split_class_data(trainset, forget_class,
                                                                                  num_forget=num_forget)
    test_forget_index, test_remain_index, _ = split_class_data(testset, forget_class, num_forget=len(testset))

    repair_class_index = random.sample(class_remain_index, int(repair_num_ratio * len(class_remain_index)))

    train_forget_sampler = SubsetRandomSampler(train_forget_index)  # 5000
    train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 45000

    repair_class_sampler = SubsetRandomSampler(repair_class_index)

    test_forget_sampler = SubsetRandomSampler(test_forget_index)  # 1000
    test_remain_sampler = SubsetRandomSampler(test_remain_index)  # 9000

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_forget_sampler)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_remain_sampler)

    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=repair_class_sampler)

    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_forget_sampler)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_remain_sampler)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index


def get_forget_loader(dt, forget_class):
    idx = []
    els_idx = []
    count = 0
    for i in range(len(dt)):
        _, lbl = dt[i]
        if lbl == forget_class:
            # if forget:
            #     count += 1
            #     if count > forget_num:
            #         continue
            idx.append(i)
        else:
            els_idx.append(i)
    forget_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(idx), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(els_idx), drop_last=True)
    return forget_loader, remain_loader


def load_parser():
    parser = argparse.ArgumentParser("Boundary Unlearning")
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='training epoch')


    parser.add_argument('--evaluation', action='store_true', help='evaluate unlearn model')
    parser.add_argument('--extra_exp', type=str, help='optional extra experiment for boundary shrink',
                        choices=['curv', 'weight_assign', None])
    
    parser.add_argument('--config', type=str, default='default.yaml', help='config name')
    parser.add_argument('--wand', type=int, default = -1)
    parser.add_argument('--wandb_tags', type=str, default = 'None')
    args = parser.parse_args()

    return args


def load_config(args):
    path = Path(os.path.realpath(__file__))
    path = str(path.parent.absolute())
    root = path + "/config/" + args.config
    with open(root) as file:
        config = yaml.safe_load(file)
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    # def convert(s):
    #     try:
    #         return float(s)
    #     except ValueError:
            
    #         return float(num) / float(denom)

    config = dotdict(config)
    # config.alpha = args.alpha if args.alpha != -1 else config.alpha
    # config.beta = args.beta if args.beta != -1 else config.beta
    # config.gamma = args.gamma if args.gamma != -1 else config.gamma
    # config.eta = args.eta if args.eta != -1 else config.eta
    # config.temperature = args.temperature if args.temperature != -1 else config.temperature
    config.epoch = args.epoch if args.epoch != -1 else config.epochs
    config.lr = args.lr if args.lr != -1 else config.lr
    config.batch_size = args.batch_size if args.batch_size != -1 else config.batch_size
    config.wand = args.wand if args.wand != -1 else config.wand
    config.wandb_tags = args.wandb_tags if args.wandb_tags != "None" else config.wandb_tags
    config.config = args.config
    

    # num, denom = config.eps.split('/')
    # config.eps = float(num)/float(denom)
    # num1,num2,num3 = config.step_size.split('/')
    # config.step_size = (float(num1)/float(num2))/float(num3)

    return config