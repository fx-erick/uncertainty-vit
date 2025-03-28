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
import numpy as np
from scipy.special import softmax
import os
import torch
import torch.distributions as dists
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from utils import load_model_for_ensembles
from scipy.stats import rankdata
from PIL import Image
import torchvision
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import torchvision.datasets as dset
import torchvision.transforms as trn
from tin import TinyImageNetC
import utils
from torchmetrics import AUROC
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

@torch.no_grad()
def evaluate_MC_dropout(data_loader, model, device, num_classes, forward_passes ):
    criterion = torch.nn.CrossEntropyLoss()
    outputs = []
    targets = []

    # switch to evaluation mode

    auroc = AUROC(task="multiclass", num_classes=num_classes)
    TACE = TACELoss()
    ECE = ECELoss()
    for i in range(forward_passes):
        model.eval()
        enable_dropout(model)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test MC Dropout:'
        temp_output = []

        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]
            images = images.to(device)
            target = target.to(device)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)

            if i == 1:
                targets.append(target)

            temp_output.append(output)

        temp_output = torch.cat(temp_output)
        outputs.append(temp_output)

    outputs = torch.stack(outputs)
    targets = torch.cat(targets)
    mean_outputs = torch.mean(outputs, 0)

    acc1, acc5 = accuracy(mean_outputs, targets, topk=(1, 5))
    val_ECE = ECE.loss(mean_outputs, targets)
    val_TACE = TACE.loss(mean_outputs, targets)
    val_NLL = NLL(mean_outputs.cpu(), targets.cpu())
    val_auroc = auroc(mean_outputs, targets)

    print(
        'MC-Dropout Acc@1 {top1:.5f} Acc@5 {top5:.5f}  ECE {ECEs:.5f} TACE {TACEs:.5f} NLL {NLLs:.5f} AUROC {AUROC:.5f} '
            .format(top1=acc1, top5=acc5, ECEs=val_ECE, TACEs=val_TACE, NLLs=val_NLL, AUROC=val_auroc))



@torch.no_grad()
# taken from https://github.com/Jonathan-Pearce/calibration_library/blob/master/metrics.py




class BrierScore():
    def __init__(self) -> None:
        pass

    def loss(self, outputs, targets):
        K = outputs.shape[1]
        one_hot = np.eye(K)[targets]
        probs = softmax(outputs, axis=1)
        return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


class CELoss(object):

    def compute_bin_boundaries(self, probabilities=np.array([])):

        # uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            # size of bins
            bin_n = int(self.n_data / self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)

            for i in range(0, self.n_bins):
                bin_boundaries = np.append(bin_boundaries, probabilities_sort[i * bin_n])
            bin_boundaries = np.append(bin_boundaries, 1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

    def get_probabilities(self, output, labels, logits):
        # If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels.detach().numpy()
        self.confidences,_ = self.probabilities.max(axis=1)
        self.predictions = np.argmax(self.probabilities, axis = 1)
        self.accuracies = np.equal(self.predictions, labels)


    def binary_matrices(self):
        idx = np.arange(self.n_data)
        # make matrices of zeros
        pred_matrix = np.zeros([self.n_data, self.n_class])
        label_matrix = np.zeros([self.n_data, self.n_class])
        # self.acc_matrix = np.zeros([self.n_data,self.n_class])

        pred_matrix[idx, self.predictions] = 1
        label_matrix[idx, self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)

    def compute_bins(self, index=None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies.detach().numpy()
        else:
            confidences = self.probabilities[:, index]
            #self.labels = self.labels.detach().numpy()
            accuracies = (self.labels == index).astype("float")

        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            a = bin_lower.item()

            bl = np.greater(confidences, bin_lower.item())
            bu = np.less_equal(confidences, bin_upper.item())
            in_bin =  bl*bu
            in_bin = in_bin.detach().numpy()
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin].detach().numpy())
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])


class MaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins=15, logits=True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()


# http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins=15, logits=True, device = "cpu"):
        super().loss(output.to(device), labels.to(device), n_bins, logits)
        return np.dot(self.bin_prop, self.bin_score)


class MCELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins=15, logits=True):
        super().loss(output, labels, n_bins, logits)
        return np.max(self.bin_score)


# https://arxiv.org/abs/1905.11001
# Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins=15, logits=True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop, self.bin_conf * np.maximum(self.bin_conf - self.bin_acc, np.zeros(self.n_bins)))


# https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):

    def loss(self, output, labels, n_bins=15, logits=True):
        sce = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            sce += np.dot(self.bin_prop, self.bin_score)

        return sce / self.n_class


class TACELoss(CELoss):

    def loss(self, output, labels, threshold=0.01, n_bins=30, logits=True, device = "cpu"):
        output = output.to(device)
        labels = labels.to(device)
        tace = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])


        super().get_probabilities(output, labels, logits)
        self.probabilities[self.probabilities < threshold] = 0
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bin_boundaries(self.probabilities[:, i])
            super().compute_bins(i)
            tace += np.dot(self.bin_prop, self.bin_score)

        return tace / self.n_class


# create TACELoss with threshold fixed at 0
class ACELoss(TACELoss):

    def loss(self, output, labels, n_bins=15, logits=True):
        return super().loss(output, labels, 0.0, n_bins, logits)

def NLL(output, target):
    output = torch.softmax(output.float(), dim = 1)
    return -dists.Categorical(output).log_prob(target).mean()

#dist transformer calculations

def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1 = torch.sigmoid(mean1)
    mean2 = torch.sigmoid(mean2)
    cov1= torch.sigmoid(cov1)
    cov2 = torch.sigmoid(cov2)

    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)
    #ret = torch.clamp(-2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2), min=1e-24)
    #ret = torch.sqrt(ret)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    #cov_ret = torch.clamp(-2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2), min=1e-24)
    #cov_ret = torch.sqrt(cov_ret)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret

def kl_distance_matmul(mean1, cov1, mean2, cov2):
    cov1_det = 1 / torch.prod(cov1, -1, keepdim=True)
    cov2_det = torch.prod(cov2, -1, keepdim=True)
    log_det = torch.log(torch.matmul(cov1_det, cov2_det.transpose(-1, -2)))

    trace_sum = torch.matmul(1 / cov2, cov1.transpose(-1, -2))

    #mean_cov_part1 = torch.matmul(mean1 / cov2, mean1.transpose(-1, -2))
    #mean_cov_part1 = torch.matmul(mean1 * mean1, (1 / cov2).transpose(-1, -2))
    #mean_cov_part2 = -torch.matmul(mean1 / cov2, mean2.transpose(-1, -2))
    #mean_cov_part2 = -torch.matmul(mean1 * mean2, (1 / cov2).transpose(-1, -2))
    #mean_cov_part3 = -torch.matmul(mean2 / cov2, mean1.transpose(-1, -2))
    #mean_cov_part4 = torch.matmul(mean2 / cov2, mean2.transpose(-1, -2))
    #mean_cov_part4 = torch.matmul(mean2 * mean2, (1 / cov2).transpose(-1, -2))

    #mean_cov_part = mean_cov_part1 + mean_cov_part2 + mean_cov_part3 + mean_cov_part4
    mean_cov_part = torch.matmul((mean1 - mean2) ** 2, (1/cov2).transpose(-1, -2))

    return (log_det + mean_cov_part + trace_sum - mean1.shape[-1]) / 2

'''@torch.no_grad()
def c_evaluate(dataset, model, device, args = None):

    header = 'Corrupted dataset evaluation :'
    # switch to evaluation mode
    errors = []
    accs = []
    for distortion_name in DISTORTIONS:
        metric_logger = utils.MetricLogger(delimiter="  ")
        print("Distortion : " + distortion_name)
        data_loader = build_c_dataset(dataset, distortion_name, args)
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]
            images = images.to(device)
            target = target.to(device)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)


            acc1, _ = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        # gather the stats from all processes

        metric_logger.synchronize_between_processes()
        print(
            '* Acc@1 {top1.global_avg:.4f} CE {CE:.4f} '
            .format(top1=metric_logger.acc1, CE=(100 -metric_logger.acc1.global_avg)/100))

        accs.append(metric_logger.acc1.global_avg)
        errors.append((100 -metric_logger.acc1.global_avg)/100)

    print('mCE (unnormalized) (%): {:.4f}, acc :{:.4f}'.format( np.mean(errors),np.mean(accs) ))'''

@torch.no_grad()
def c_evaluate(dataset, model, device, args = None):

    header = 'Corrupted dataset evaluation :'
    # switch to evaluation mode
    errors = []
    accs = []
    for distortion_name in DISTORTIONS:
        metric_logger = utils.MetricLogger(delimiter="  ")
        print("Distortion : " + distortion_name)
        for severity in range(1, 6):
            data_loader = build_c_dataset( distortion_name, severity, args)
            for batch in metric_logger.log_every(data_loader, 10, header):
                images = batch[0]
                target = batch[-1]
                images = images.to(device)
                target = target.to(device)

                # compute output
                with torch.cuda.amp.autocast():
                    output = model(images)


                acc1, _ = accuracy(output, target, topk=(1, 5))
                batch_size = images.shape[0]
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

            # gather the stats from all processes

            metric_logger.synchronize_between_processes()
            print(
                '* Acc@1 {top1.global_avg:.4f} CE {CE:.4f} '
                .format(top1=metric_logger.acc1, CE=(100 -metric_logger.acc1.global_avg)/100))

            accs.append(metric_logger.acc1.global_avg)
            errors.append((100 -metric_logger.acc1.global_avg)/100)

    print('mCE (unnormalized) (%): {:.4f}, acc :{:.4f}'.format( np.mean(errors),np.mean(accs) ))

@torch.no_grad()
def ensembles_c_evaluate(dataset, model, device, args = None):
    header = 'Corrupted dataset evaluation :'
    # switch to evaluation mode
    errors = []
    accs = []
    filename = "output/checkpoint-ensembles-/checkpoint-14/mp_rank_00_model_states.pt"

    for distortion_name in DISTORTIONS:
        metric_logger = utils.MetricLogger(delimiter="  ")
        print("Distortion : " + distortion_name)
        outputs = []
        targets = []
        # switch to evaluation mode
        data_loader = build_c_dataset(dataset, distortion_name, args)


        for i in range(1, 6):
            metric_logger = utils.MetricLogger(delimiter="  ")
            header = 'Test Ensembles Corrupt:'
            f = filename[:28] + str(i) + filename[28:]
            print(f)
            model = load_model_for_ensembles(model, f, device, args)


            temp_output = []
            for batch in metric_logger.log_every(data_loader, 10, header):
                images = batch[0]
                target = batch[-1]
                images = images.to(device)
                target = target.to(device)

                # compute output
                with torch.cuda.amp.autocast():
                    output = model(images)

                acc1, _ = accuracy(output, target, topk=(1, 5))

                batch_size = images.shape[0]
                temp_output.append(output)
                if i == 1:
                    targets.append(target)

                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            temp_output = torch.cat(temp_output)
            outputs.append(temp_output)


        metric_logger = utils.MetricLogger(delimiter="  ")

        outputs = torch.stack(outputs)
        mean_outputs = torch.mean(outputs, 0)


        id = 0
        for batch in metric_logger.log_every(data_loader, 10, header):
            target = batch[-1]

            target = target.to(device)
            batch_size = target.shape[0]

            # compute output
            with torch.cuda.amp.autocast():
                output = mean_outputs[id:id + batch_size]


            acc1, _ = accuracy(output, target, topk=(1, 5))
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)


            id = id + batch_size

        metric_logger.synchronize_between_processes()
        print(
            'Ensembles Distortion {} : Acc@1 {top1.global_avg:.5f} CE {CE:.5f}'
                .format(distortion_name, top1=metric_logger.acc1, CE=(100 -metric_logger.acc1.global_avg)/100))

        accs.append(metric_logger.acc1.global_avg)
        errors.append((100 - metric_logger.acc1.global_avg) / 100)

    print('mCE (unnormalized) (%): {:.4f}, acc :{:.4f}'.format(np.mean(errors), np.mean(accs)))


@torch.no_grad()
def mc_dropout_c_evaluate(dataset, model, device, forward_passes, args):
    header = 'Corrupted dataset evaluation :'
    # switch to evaluation mode
    errors = []
    accs = []

    for distortion_name in DISTORTIONS:

        metric_logger = utils.MetricLogger(delimiter="  ")
        print("Distortion : " + distortion_name)
        dropout_errors= []
        dropout_accs = []
        # switch to evaluation mode
        data_loader = build_c_dataset(dataset, distortion_name, args)


        for i in range(forward_passes):
            model.eval()
            enable_dropout(model)
            metric_logger = utils.MetricLogger(delimiter="  ")
            header = 'Test MC Dropout Corrupt:'

            for batch in metric_logger.log_every(data_loader, 10, header):
                images = batch[0]
                target = batch[-1]
                images = images.to(device)
                target = target.to(device)

                # compute output
                with torch.cuda.amp.autocast():
                    output = model(images)

                acc1, _ = accuracy(output, target, topk=(1, 5))

                batch_size = images.shape[0]

                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

            # gather the stats from all processes
            dropout_accs.append(metric_logger.acc1.global_avg)
            dropout_errors.append((100 - metric_logger.acc1.global_avg) / 100)




        distortion_acc = np.mean(dropout_accs)
        distortion_err = np.mean(dropout_errors)
        print(
            'MC Distortion {} : Acc@1 {top1:.5f} CE {CE:.5f}'
                .format(distortion_name, top1=distortion_acc, CE=distortion_err))

        accs.append(distortion_acc)
        errors.append(distortion_err)

    print('mCE (unnormalized) (%): {:.4f}, acc :{:.4f}'.format(np.mean(errors), np.mean(accs)))


@torch.no_grad()
def dist_c_evaluate(dataset, model, device, args = None):
    header = 'Corrupted dataset evaluation :'

    # switch to evaluation mode
    errors = []
    accs = []
    for distortion_name in DISTORTIONS:
        metric_logger = utils.MetricLogger(delimiter="  ")
        print("Distortion : " + distortion_name)
        data_loader = build_c_dataset(dataset, distortion_name, args)
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]

            images = images.to(device)
            target = target.to(device)


            # compute output
            with torch.cuda.amp.autocast():
                _, _, output = model(images)

            acc1, _ = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} CE {CE:.3f} '.format(top1=metric_logger.acc1, CE=(100 -metric_logger.acc1.global_avg)/100))
        accs.append(metric_logger.acc1.global_avg)
        errors.append((100 - metric_logger.acc1.global_avg) / 100)

    print('mCE (unnormalized) (%): {:.4f}, acc :{:.4f}'.format(np.mean(errors), np.mean(accs)))

'''def build_c_dataset(dataset, distortion, args):
    data_path = args.data_path[:-15] + 'cifar-100-c'
    data_pth = os.path.join(data_path, f"{distortion}.npy")
    labels_pth = os.path.join(data_path, "labels.npy")

    dataset.data = np.load(data_pth)
    dataset.targets = torch.LongTensor(np.load(labels_pth))

    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return data_loader'''




def build_c_dataset(distortion, severity, args):
    data_path = args.data_path + '/' +  str(distortion) + '/' + str(severity)
    print(data_path)
    dataset = TinyImageNetC( root_dir= data_path,
            transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]))

    #sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return data_loader


@torch.no_grad()
def p_evaluate(dataset,model, device, args = None):

    print('Perturbed dataset evaluation :')
    # switch to evaluation mode
    num_classes = args.nb_classes
    flip_list = []
    zipf_list = []
    for p in PERTURBATIONS:
        print("Perturbation : " + p)
        data_loader = build_p_dataset( dataset, p, args)
        predictions, ranks = [], []

        for raw_data in data_loader:
            num_vids = raw_data.size(0)
            data = process_raw_data(raw_data, device)
            data.to(device)

            with torch.cuda.amp.autocast():
                if args.stochastic:
                    _,_,output = model(data)
                elif args.ensembles:
                    output = ensembles_p_evaluate(model, data, device, args)
                else:
                    output = model(data)

            for vid in output.view(num_vids, -1, num_classes):
                predictions.append(vid.argmax(1).to('cpu').numpy())
                ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])

        ranks = np.asarray(ranks)

        print('\nComputing Metrics for', p,)

        current_flip = flip_prob(predictions, True if 'noise' in p else False)
        current_zipf = ranking_dist(ranks, True if 'noise' in p else False, mode='zipf')
        flip_list.append(current_flip)
        zipf_list.append(current_zipf)

        print('\n' + p, 'Flipping Prob')
        print(current_flip)
        print('Top5 Distance\t{:.5f}'.format(ranking_dist(ranks, True if 'noise' in p else False, mode='top5')))
        print('Zipf Distance\t{:.5f}'.format(current_zipf))

    print(flip_list)
    print('\nMean Flipping Prob\t{:.5f}'.format(np.mean(flip_list)))


@torch.no_grad()
def mc_dropout_p_evaluate(dataset,model, device, args = None):
    print('Perturbed dataset evaluation :')
    # switch to evaluation mode
    num_classes = args.nb_classes
    flip_list = []
    zipf_list = []

    for p in PERTURBATIONS:

        print("Perturbation : " + p)
        data_loader = build_p_dataset(dataset, p, args)
        predictions, ranks = [], []
        outputs = []

        for i in range(args.mc_dropout_forwards):
            model.eval()
            enable_dropout(model)
            temp_output = []
            for raw_data in data_loader:
                num_vids = raw_data.size(0)
                data = process_raw_data(raw_data, device)
                data.to(device)

                with torch.cuda.amp.autocast():
                    output = model(data)

                temp_output.append(output)
            temp_output = torch.cat(temp_output)
            outputs.append(temp_output)
        outputs = torch.stack(outputs)
        mean_outputs = torch.mean(outputs, 0)

        index = 0
        for raw_data in data_loader:
            num_vids = raw_data.size(0)
            out = mean_outputs[index:index+num_vids]

            for vid in out.view(num_vids, -1, num_classes):
                predictions.append(vid.argmax(1).to('cpu').numpy())
                ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])

            index = index + num_vids

        ranks = np.asarray(ranks)

        print('\nComputing Metrics for', p, )

        current_flip = flip_prob(predictions, True if 'noise' in p else False)
        current_zipf = ranking_dist(ranks, True if 'noise' in p else False, mode='zipf')
        flip_list.append(current_flip)
        zipf_list.append(current_zipf)

        print('\n' + p, 'Flipping Prob')
        print(current_flip)
        print('Top5 Distance\t{:.5f}'.format(ranking_dist(ranks, True if 'noise' in p else False, mode='top5')))
        print('Zipf Distance\t{:.5f}'.format(current_zipf))

    print(flip_list)
    print('\nMean Flipping Prob\t{:.5f}'.format(np.mean(flip_list)))


@torch.no_grad()
def ensembles_p_evaluate(model, data, device, args = None):

    filename = "output/checkpoint-ensembles-/checkpoint-14/mp_rank_00_model_states.pt"
    outputs = []

    for i in range(1, 6):
        f = filename[:28] + str(i) + filename[28:]
        model = load_model_for_ensembles(model, f, device, args)
        output = model(data)
        outputs.append(output)

    outputs = torch.stack(outputs)
    mean_outputs = torch.mean(outputs, 0)
    return mean_outputs





def ranking_dist(ranks, noise_perturbation=False, mode='top5'):
    result = 0
    step_size = 1

    for vid_ranks in ranks:
        result_for_vid = []

        for i in range(step_size):
            perm1 = vid_ranks[i]
            perm1_inv = np.argsort(perm1)

            for rank in vid_ranks[i::step_size][1:]:
                perm2 = rank
                result_for_vid.append(dist(perm2[perm1_inv], mode))
                if not noise_perturbation:
                    perm1 = perm2
                    perm1_inv = np.argsort(perm1)

        result += np.mean(result_for_vid) / len(ranks)

    return result


def flip_prob(predictions, noise_perturbation=False):
    result = 0
    step_size = 1

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation: prev_pred = pred

        result += np.mean(result_for_vid) / len(predictions)

    return result

def build_p_dataset( dataset, perturbation, args):
    data_path = args.data_path[:-15] + 'cifar-100-p'
    data_pth = os.path.join(data_path, f"{perturbation}.npy")

    dataset = torch.from_numpy(np.float32(np.load(data_pth).transpose((0, 1, 4, 2, 3)))) / 255.
    #ood_data = torch.utils.data.TensorDataset(dataset, dummy_targets)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return data_loader

def process_raw_data(raw_data, device):
    raw_data = raw_data.view(-1, 3, 32, 32).cuda()
    num_images = raw_data.shape[0]

    tensor_list = []
    for i in range(num_images):
        if num_images > 1:
            image = raw_data[i]
        else:
            image = raw_data
        image = torchvision.transforms.ToPILImage()(image)

        image = torchvision.transforms.Resize(256,interpolation=3)(image)
        image = torchvision.transforms.CenterCrop(224)(image)

        image_tensor = torchvision.transforms.ToTensor()(image).to(device)

        mean = IMAGENET_INCEPTION_MEAN
        std = IMAGENET_INCEPTION_STD
        image_tensor = torchvision.transforms.Normalize(mean, std)(image_tensor).to(device)
        tensor_list.append(image_tensor)

    data = torch.stack(tensor_list, dim=0) * 2 - 1
    data = data.half()

    return data


def dist(sigma, mode='top5'):
    num_classes = 100
    identity = np.asarray(range(1, num_classes + 1))
    cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (num_classes - 1 - 5)))
    recip = 1. / identity
    if mode == 'top5':
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
    elif mode == 'zipf':
        return np.sum(np.abs(recip - recip[sigma-1])*recip)

'''DISTORTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                   'defocus_blur', 'glass_blur', 'motion_blur',
                   'zoom_blur', 'snow', 'frost',
                   'brightness', 'contrast', 'elastic_transform',
                   'pixelate', 'jpeg_compression', 'speckle_noise',
                   'gaussian_blur', 'spatter', 'saturate']'''

DISTORTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                   'defocus_blur', 'glass_blur', 'motion_blur',
                   'zoom_blur', 'snow', 'frost',
                   'brightness', 'contrast', 'elastic_transform',
                   'pixelate', 'jpeg_compression', 'speckle_noise']

PERTURBATIONS = ['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
          'snow', 'brightness', 'translate', 'rotate', 'tilt', 'scale']
