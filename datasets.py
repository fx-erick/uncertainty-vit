# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
# Copyright (c) Meta Platforms, Inc. and affiliates
import os
import torch

from torchvision import datasets, transforms
from tin import TinyImageNetDataset
from tin_2 import TinyImageNet

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic
from timm.data import create_transform

from dall_e.utils import map_pixels
from masking_generator import MaskingGenerator
from dataset_folder import ImageFolder
from PIL import Image
from cifar_semi import x_u_split, CIFAR100SSL


class DataAugmentationForBEiT(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if args.aug_level == 0:
            print(' >>>>>> args.aug_level', args.aug_level)
            self.common_transform = transforms.Compose([
                transforms.CenterCrop(size=args.input_size)
            ])
        elif args.aug_level == 1:
            print(' >>>>>> args.aug_level', args.aug_level)
            self.common_transform = transforms.Compose([
                transforms.Resize(size=int(args.input_size / .875), interpolation=Image.BICUBIC),
                transforms.CenterCrop(size=args.input_size)
            ])
        elif args.aug_level == 2:
            print(' >>>>>> args.aug_level', args.aug_level)
            self.common_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(size=int(args.input_size / .875), interpolation=Image.BICUBIC),
                transforms.CenterCrop(size=args.input_size)
            ])
        elif args.aug_level == 3:
            print(' >>>>>> args.aug_level', args.aug_level)
            self.common_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=args.input_size, interpolation=Image.BICUBIC)
            ])
        elif args.aug_level == 4:
            print(' >>>>>> args.aug_level', args.aug_level)
            self.common_transform = transforms.Compose([
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=args.input_size, interpolation=Image.BICUBIC)
            ])
        else:
            self.common_transform = transforms.Compose([
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=getattr(args, 'second_input_size', None),
                    interpolation=args.train_interpolation, second_interpolation=getattr(args, 'second_interpolation', None),
                ),
            ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        if getattr(args, 'discrete_vae_type', None) is None:
            self.visual_token_transform = lambda z: z
        elif args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        z = self.common_transform(image)
        if isinstance(z, tuple):
            for_patches, for_visual_tokens = z
            return \
                self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
                self.masked_position_generator()
        else:
            return self.patch_transform(z), self.masked_position_generator()


    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_beit_pretraining_dataset(args):
    transform = DataAugmentationForBEiT(args)
    print("Data Aug = %s" % str(transform))
    if args.data_set == 'CIFAR100':
        return datasets.CIFAR100(args.data_path, train=True, transform=transform)
    elif args.data_set == 'CIFAR10':
        return datasets.CIFAR10(args.data_path, train = True, transform = transform)
    else:
        return ImageFolder(args.data_path, transform=transform)


def build_dataset(is_train, args):
    '''if args.stochastic :
        transform = DataAugmentationForBEiT(args)
    else:
        transform = build_transform(is_train, args)'''
    transform = build_transform(is_train, args)
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

        if is_train and args.semi_supervised_ratio > 0:

            dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
            labels = dataset.targets
            total_labels = len(dataset)
            num_labeled = int(50000 * args.semi_supervised_ratio)
            print(f"Performing semi-supervised training with {num_labeled} labels out of {total_labels}")
            train_labeled_idxs, train_unlabeled_idxs = x_u_split(num_labeled, nb_classes, labels)
            dataset = CIFAR100SSL(
                args.data_path, train_labeled_idxs, train=True,
                transform=transform)
            print(f"Length of the semi-supervised dataset {len(dataset)}")
        else:
            dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
    elif args.data_set == 'CIFAR10' or args.data_set == 'CIFAR10-C':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform = transform )
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform, is_valid_file=is_valid_file)
        nb_classes = 1000
    elif args.data_set == 'tiny_IMNET' :
        root = args.data_path
        train_mode = 'train' if is_train else 'val'
        dataset = TinyImageNetDataset(root, mode = train_mode ,  transform=transform)
        nb_classes = 200
    elif args.data_set == 'SVHN':
        split = 'train' if is_train else "test"
        root = os.path.join(args.data_path, split)
        dataset = datasets.SVHN(root, split = split, transform = transform)
        nb_classes = 10
    elif args.data_set == "image_folder" or args.data_set == 'tiny_IMNET-C':
        if args.data_set == 'tiny_IMNET-C':
            root = args.data_path + '/gaussian_noise/1'
        else:
            root = args.data_path if is_train else args.eval_data_path

        dataset = ImageFolder(root, transform=transform, is_valid_file=is_valid_file)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes, f"{nb_classes} != {args.nb_classes}"
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


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
