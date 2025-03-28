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
import math
import sys
from typing import Iterable
import argparse
import torch
import torch.nn.functional as F
from distloss import WassersteinLoss, WassersteinLossFineTuning
from timm.models import create_model
from timm.utils import ModelEmaV2
from optim_factory import create_optimizer
import modeling_finetune
from utils import NativeScalerWithGradNormCount as NativeScaler

import utils

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets import build_dataset
import dist_datasets
from engine_for_finetuning import train_one_epoch, evaluate
from uncertainty_evaluations import evaluate_MC_dropout
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from scipy import interpolate
from distloss import WassersteinLossFineTuning
from uncertainty_evaluations import ECELoss, TACELoss, NLL
from timm.utils import accuracy

from torchmetrics import AUROC

def get_args():
    parser = argparse.ArgumentParser('BEiT fine-tuning and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Model parameters
    parser.add_argument('--model', default='dist_beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default= 1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)
    parser.add_argument('--target_layer', default=-1, type=int, help="target output layer (0-based)")
    parser.add_argument('--remove_final_norm', action='store_true', dest='remove_final_norm')
    parser.add_argument('--reinit_final_norm', action='store_true', dest='reinit_final_norm')
    parser.add_argument('--learn_layer_weights', action='store_true', dest='learn_layer_weights')  # supersede `target_layer`
    parser.add_argument('--layernorm_before_combine', action='store_true', dest='layernorm_before_combine')

    # Dataset parameters
    parser.add_argument('--data_path', default=r'C:\Users\erickfs\PycharmProjects\MA\AugmentedCIFAR', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')
    parser.add_argument('--linear_classifier', action='store_true',
                        help='linear classifier')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--data_set', default='CIFAR', choices=['CIFAR', 'IMNET', 'image_folder', 'tiny_IMNET'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--data_set_filter_file', type=str, default=None, help="path to filter to filter dataset")
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    parser.add_argument(
        "--num_mask_patches",
        default=0,
        type=int,
        help="number of the visual tokens/patches need be masked",
    )
    parser.add_argument("--max_mask_patches_per_block", type=int, default=None)
    parser.add_argument("--min_mask_patches_per_block", type=int, default=16)

    parser.add_argument("--mc_dropout_forwards", type=int, default=0)
    parser.add_argument('--gp_layer', action='store_true')
    parser.add_argument('--het_layer', action='store_true')
    parser.add_argument('--sinkformer', action='store_true')
    parser.add_argument("--gumbel_softmax", default=False, action="store_true")
    parser.add_argument("--laplace", default=False, action="store_true")
    parser.add_argument('--h_sto_trans', default=False, action='store_true')
    parser.add_argument('--sngp', default=False, action='store_true')
    parser.add_argument('--stochastic', default=False, action='store_true')
    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:

            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args()




def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_rate=args.drop,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        attn_drop_rate=args.attn_drop_rate,
        gp_layer=args.gp_layer,
        gumbel_softmax=args.gumbel_softmax,
        sinkformer=args.sinkformer,
        h_sto_trans=args.h_sto_trans
    )

    return model


def train_class_batch(model, samples, pos_targets, neg_targets, labels, criterion, dist_criterion,
                      bool_masked_pos=None):
    mean_outputs, cov_outputs, outputs = model(samples, bool_masked_pos=bool_masked_pos)
    loss = criterion(outputs, labels)

    import copy
    '''if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        dummy_model = copy.deepcopy(model.module)
    else:
        dummy_model = copy.deepcopy(model)'''
    dummy_model = copy.deepcopy(model.module)
    dummy_model.eval()
    mean_pos, cov_pos, _ = dummy_model(pos_targets, bool_masked_pos=bool_masked_pos)
    mean_neg, cov_neg, _ = dummy_model(neg_targets, bool_masked_pos=bool_masked_pos)
    w_loss = dist_criterion(mean_outputs, cov_outputs, mean_pos, cov_pos, mean_neg, cov_neg)

    loss = loss + w_loss

    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def dist_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, dist_criterion: torch.nn.Module,
                               data_loader: Iterable, optimizer: torch.optim.Optimizer,
                               device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                               model_ema=None, mixup_fn=None, log_writer=None,
                               start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                               num_training_steps_per_epoch=None, update_freq=None, masked_position_generator=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, neg_targets, labels) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None and 'lr_scale' in param_group:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if lr_schedule_values is not None and 'lr_scale' not in param_group:
                    param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        bool_masked_pos = None
        if masked_position_generator is not None:
            bool_masked_pos = torch.tensor([masked_position_generator() for _ in range(samples.size(0))], device=device)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        neg_targets = neg_targets.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        if mixup_fn is not None:
            samples, labels = mixup_fn(samples, labels)

        if loss_scaler is None:
            samples, targets, neg_targets = samples.half(), targets.half(), neg_targets.half()
            loss, output = train_class_batch(
                model, samples, targets, neg_targets, labels, criterion, dist_criterion, bool_masked_pos)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, neg_targets, labels, criterion, dist_criterion, bool_masked_pos)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            #class_acc = (output.max(-1)[-1] == targets).float().mean()
            class_acc = None
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def dist_evaluate(data_loader, model, device, num_classes, dist_criterion):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        pos_image = batch[1]
        neg_image = batch[2]
        images = images.to(device)
        target = target.to(device)
        pos_image = pos_image.to(device)
        neg_image = neg_image.to(device)

        # compute output
        with torch.cuda.amp.autocast():
            mean_outputs, cov_outputs, output = model(images)
            loss = criterion(output, target)
            mean_pos, cov_pos, _ = model(pos_image)
            mean_neg, cov_neg, _ = model(neg_image)
            w_loss = dist_criterion(mean_outputs, cov_outputs, mean_pos, cov_pos, mean_neg, cov_neg)
            loss = loss + w_loss

        auroc = AUROC(task="multiclass", num_classes=num_classes)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # val_ECE = ECE(target, output, device = device)
        ECE = ECELoss()
        TACE = TACELoss()
        val_ECE = ECE.loss(output, target)
        val_TACE = TACE.loss(output, target)
        val_NLL = NLL(output.cpu(), target.cpu())
        val_auroc = auroc(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['ECE'].update(val_ECE.item(), n=batch_size)
        metric_logger.meters['TACE'].update(val_TACE.item(), n=batch_size)
        metric_logger.meters['NLL'].update(val_NLL.item(), n=batch_size)
        metric_logger.meters['AUROC'].update(val_auroc.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} ECE {ECEs.global_avg:.3f} TACE {TACEs.global_avg:.3f} NLL {NLLs.global_avg:.3f} AUROC {AUROC.global_avg:.3f} '
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, ECEs=metric_logger.ECE,
                TACEs=metric_logger.TACE, NLLs=metric_logger.NLL, AUROC=metric_logger.AUROC))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args()


    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    dataset_train, args.nb_classes = dist_datasets.build_dataset(is_train=True, args=args)
    dataset_val, _ = dist_datasets.build_dataset(is_train=False, args=args)
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        linear_classifier=args.linear_classifier,
        has_masking=args.num_mask_patches > 0,
        learn_layer_weights=args.learn_layer_weights,
        layernorm_before_combine=args.layernorm_before_combine,
        gp_layer=args.gp_layer,
        het_layer=args.het_layer,
        sinkformer=args.sinkformer,
        gumbel_softmax=args.gumbel_softmax,
        h_sto_trans=args.h_sto_trans,
        sngp=args.sngp
    )
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    device = torch.device("cpu")
    epoch = 1


    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    total_batch_size = args.batch_size * args.update_freq * 1
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size




    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    criterion = SoftTargetCrossEntropy()
    dist_criterion = WassersteinLossFineTuning()
    model_ema = None
    '''train_stats = dist_train_one_epoch(
        model, criterion, dist_criterion, data_loader_train, optimizer,
        device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
        log_writer=None, start_steps=epoch * num_training_steps_per_epoch,
        lr_schedule_values=None, wd_schedule_values=None,
        num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
        masked_position_generator=None,
    )'''
    test_stats = dist_evaluate(data_loader_val, model, device)
    # mean_x,cov_x = model(a, b)