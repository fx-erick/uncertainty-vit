import os
import torch
import copy
from torchvision import datasets, transforms
import numpy as np

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic
from timm.data import create_transform

from PIL import Image
from cifar_semi import x_u_split, AugmentedCIFAR100SSL
import random
from tin import TinyImageNetDataset
import imageio
import tin


class AugmentedCIFAR100(datasets.CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        download: bool = False,
    ) :
        super().__init__(root, train =  train, transform=transform, download = download, target_transform=target_transform)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        neg_img = neg_sample(target, self.data, self.targets)
        img, neg_img = Image.fromarray(img), Image.fromarray(neg_img)

        pos_img = copy.deepcopy(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            pos_img = self.target_transform(pos_img)
            neg_img = self.transform(neg_img)

        return img, pos_img, neg_img, target


class AugmentedCIFAR10(datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        download: bool = False,
    ) :
        super().__init__(root, train =  train, transform=transform, download = download, target_transform=target_transform)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        neg_img = neg_sample(target, self.data, self.targets)
        img, neg_img = Image.fromarray(img), Image.fromarray(neg_img)

        pos_img = copy.deepcopy(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            pos_img = self.target_transform(pos_img)
            neg_img = self.transform(neg_img)

        return img, pos_img, neg_img, target


class AugmentedSVHN(datasets.SVHN):
    def __init__(
        self,
        root: str,
        split : str,
        transform = None,
        target_transform = None,
        download: bool = False,
    ) :
        super().__init__(root, split = split, transform=transform, download = download, target_transform=target_transform)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.labels[index])
        neg_img = neg_sample(target, self.data, self.labels)
        img, neg_img = Image.fromarray(np.transpose(img, (1, 2, 0))), Image.fromarray(np.transpose(neg_img, (1, 2, 0)))
        pos_img = copy.deepcopy(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            pos_img = self.target_transform(pos_img)
            neg_img = self.transform(neg_img)

        return img, pos_img, neg_img, target


class AugmentedtinyIMNET(TinyImageNetDataset):
    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        transform = None,
        target_transform = None,
        download: bool = False,
        preload : bool = False,
        max_samples = None
    ) :
        super().__init__(root_dir, mode = mode, transform=transform, target_transform=target_transform)

    def __getitem__(self, index: int):

        s = self.samples[index]
        img = imageio.imread(s[0])
        img = tin._add_channels(img)
        img = Image.fromarray(img, mode="RGB")
        lbl = s[self.label_idx]

        
        neg_label = lbl 
        while neg_label == lbl:
            neg_index = random.randint(1, self.samples_num - 1)
            neg_s = self.samples[neg_index]
            neg_label = neg_s[self.label_idx]

        
        neg_img = imageio.imread(neg_s[0])
        neg_img = tin._add_channels(neg_img)
        neg_img = Image.fromarray(neg_img, mode="RGB")
        
        pos_img = copy.deepcopy(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            pos_img = self.target_transform(pos_img)
            neg_img = self.transform(neg_img)

        return img, pos_img, neg_img, lbl

def neg_sample(target, data_list, target_list):
    neg_index = random.randint(1, len(target_list) - 1)
    while target_list[neg_index] == target:
        neg_index = random.randint(1, len(target_list) - 1)

    return data_list[neg_index]


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    target_transform = build_transform(False, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    is_valid_file = None

    if is_train:
        file_filter = getattr(args, "data_set_filter_file", None)
        if file_filter is not None:
            files = set()
            with open(file_filter) as ff:
                for l in ff:
                    files.add(l.rstrip())
            is_valid_file = lambda p: os.path.basename(p) in files

    if args.data_set == 'CIFAR100' or args.data_set == 'CIFAR100-C' or args.data_set == 'CIFAR100-P':
        nb_classes = 100
        if is_train and args.semi_supervised_ratio != 0:
            dataset = AugmentedCIFAR100(args.data_path, train=is_train, transform=transform,
                                        target_transform=target_transform)
            labels = dataset.targets
            num_labeled = int(50000 * args.semi_supervised_ratio)
            train_labeled_idxs, train_unlabeled_idxs = x_u_split(num_labeled, nb_classes, labels)
            dataset = AugmentedCIFAR100SSL(
                args.data_path, train_labeled_idxs, train=True,
                transform=transform,
                target_transform=target_transform)
        else:
            dataset = AugmentedCIFAR100(args.data_path, train=is_train, transform=transform,
                                        target_transform=target_transform)



    elif args.data_set == "CIFAR10" or args.data_set =='CIFAR10-C':
        dataset = AugmentedCIFAR10(args.data_path, train=is_train, transform=transform,
                                    target_transform=target_transform)
        nb_classes = 10
    elif args.data_set == "tiny_IMNET" or args.data_set == 'tiny_IMNET-c':
        mode = 'train' if is_train else "val"
        dataset = AugmentedtinyIMNET(args.data_path, mode = mode, transform=transform,
                                    target_transform=target_transform)
        nb_classes = 200
    else :
        split = 'train' if is_train else "test"
        #split = 'train'
        root = os.path.join(args.data_path, split)
        dataset = AugmentedSVHN(root,split, transform = transform, target_transform = target_transform )
        nb_classes = 10



    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes