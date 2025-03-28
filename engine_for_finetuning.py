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
from typing import Iterable, Optional
from uncertainty_evaluations import ECELoss, TACELoss, NLL


import torch
import torch.nn.functional as F


from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from torchmetrics import AUROC
from optim_factory import create_optimizer, LayerDecayValueAssigner
from utils import NativeScalerWithGradNormCount as NativeScaler

#from torchmetrics.functional import calibration_error


import utils


def train_class_batch(model, samples, target, criterion, bool_masked_pos=None):
    outputs = model(samples, bool_masked_pos=bool_masked_pos)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
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

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None and 'lr_scale' in param_group :
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

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, bool_masked_pos)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion, bool_masked_pos)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
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
            class_acc = (output.max(-1)[-1] == targets).float().mean()
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
def evaluate(data_loader, model, device, num_classes):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode


    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device)
        target = target.to(device)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        auroc = AUROC(task="multiclass", num_classes=num_classes)

        #val_ECE = ECE(target, output, device = device)
        ECE = ECELoss()
        TACE = TACELoss()
        val_ECE = ECE.loss(output,target)
        val_TACE = TACE.loss(output,target)
        val_NLL = NLL(output.cpu(), target.cpu())
        val_auroc = auroc(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['ECE'].update(val_ECE.item(),n=batch_size)
        metric_logger.meters['TACE'].update(val_TACE.item(), n=batch_size)
        metric_logger.meters['NLL'].update(val_NLL.item(), n=batch_size)
        metric_logger.meters['AUROC'].update(val_auroc.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} ECE {ECEs.global_avg:.3f} TACE {TACEs.global_avg:.3f} NLL {NLLs.global_avg:.3f} AUROC {AUROC.global_avg:.3f} '
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, ECEs = metric_logger.ECE, TACEs = metric_logger.TACE, NLLs = metric_logger.NLL, AUROC =metric_logger.AUROC))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def ensembles_evaluate(data_loader, model, device, num_classes, args = None):
    criterion = torch.nn.CrossEntropyLoss()


    outputs = []
    targets = []
    # switch to evaluation mode


    filename = "output/checkpoint-tiny-imagenet-ensembles-/checkpoint-49/mp_rank_00_model_states.pt"

    for i in range(2,6):
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test Ensembles:'
        #f = filename[:28] + str(i) + filename[28:] CIFAR100
        #f = filename[:37] + str(i) + filename[37:] CIFAR10
        f = filename[:42] + str(i) + filename[42:]
        model = utils.load_model_for_ensembles(model, f , device, args)

        temp_output = []
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]
            images = images.to(device)
            target = target.to(device)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            auroc = AUROC(task="multiclass", num_classes=num_classes)

            #val_ECE = ECE(target, output, device = device)
            ECE = ECELoss()
            TACE = TACELoss()
            val_ECE = ECE.loss(output,target)
            val_TACE = TACE.loss(output,target)
            val_NLL = NLL(output.cpu(), target.cpu())
            val_auroc = auroc(output, target)

            batch_size = images.shape[0]
            temp_output.append(output)
            if i == 2:
                targets.append(target)
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['ECE'].update(val_ECE.item(),n=batch_size)
            metric_logger.meters['TACE'].update(val_TACE.item(), n=batch_size)
            metric_logger.meters['NLL'].update(val_NLL.item(), n=batch_size)
            metric_logger.meters['AUROC'].update(val_auroc.item(), n=batch_size)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        temp_output = torch.cat(temp_output)
        outputs.append(temp_output)
        print('* Acc@1 {top1.global_avg:.5f} Acc@5 {top5.global_avg:.5f} loss {losses.global_avg:.5f} ECE {ECEs.global_avg:.5f} TACE {TACEs.global_avg:.5f} NLL {NLLs.global_avg:.5f} AUROC {AUROC.global_avg:.5f} '
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, ECEs = metric_logger.ECE, TACEs = metric_logger.TACE, NLLs = metric_logger.NLL, AUROC =metric_logger.AUROC))

    metric_logger = utils.MetricLogger(delimiter="  ")

    outputs = torch.stack(outputs)
    targets = torch.cat(targets)
    mean_outputs = torch.mean(outputs, 0)

    acc1, acc5 = accuracy(mean_outputs, targets, topk=(1, 5))
    val_ECE = ECE.loss(mean_outputs, targets)
    val_TACE = TACE.loss(mean_outputs, targets)
    val_NLL = NLL(mean_outputs.cpu(), targets.cpu())
    val_auroc = auroc(mean_outputs, targets)

    print(
        'Ensembles Acc@1 {top1:.5f} Acc@5 {top5:.5f}  ECE {ECEs:.5f} TACE {TACEs:.5f} NLL {NLLs:.5f} AUROC {AUROC:.5f} '
        .format(top1=acc1, top5=acc5, ECEs=val_ECE, TACEs=val_TACE, NLLs=val_NLL, AUROC=val_auroc))
    id = 0
    for batch in metric_logger.log_every(data_loader, 10, header):

        target = batch[-1]

        target = target.to(device)
        batch_size = target.shape[0]

        # compute output
        with torch.cuda.amp.autocast():
            output = mean_outputs[id:id+batch_size]
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        auroc = AUROC(task="multiclass", num_classes=num_classes)

        # val_ECE = ECE(target, output, device = device)
        ECE = ECELoss()
        TACE = TACELoss()
        val_ECE = ECE.loss(output, target)
        val_TACE = TACE.loss(output, target)
        val_NLL = NLL(output.cpu(), target.cpu())
        val_auroc = auroc(output, target)




        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['ECE'].update(val_ECE.item(), n=batch_size)
        metric_logger.meters['TACE'].update(val_TACE.item(), n=batch_size)
        metric_logger.meters['NLL'].update(val_NLL.item(), n=batch_size)
        metric_logger.meters['AUROC'].update(val_auroc.item(), n=batch_size)

        id = id + batch_size

    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.5f} Acc@5 {top5.global_avg:.5f} loss {losses.global_avg:.5f} ECE {ECEs.global_avg:.5f} TACE {TACEs.global_avg:.5f} NLL {NLLs.global_avg:.5f} AUROC {AUROC.global_avg:.5f} '
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, ECEs=metric_logger.ECE,
                TACEs=metric_logger.TACE, NLLs=metric_logger.NLL, AUROC=metric_logger.AUROC))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

'''def load_model_for_ensembles(model, f, device, args):
    model_key = 'model|module'
    model = model.module
    checkpoint = torch.load(f, map_location='cpu')
    for model_key in model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    checkpoint_model = load_keys(checkpoint_model, model)
    utils.load_state_dict(model, checkpoint_model)
    model.to(device)

    skip_weight_decay_list = model.no_weight_decay()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module

    num_layers = model_without_ddp.get_num_layers()

    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler()
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=None)

def load_keys(checkpoint_model,model):
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()

            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                print("Position interpolate for %s from %dx%d to %dx%d" % (
                    key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print("Original positions = %s" % str(x))
                print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]

        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
    return checkpoint_model
'''
