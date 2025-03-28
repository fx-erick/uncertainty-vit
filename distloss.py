import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import utils

class WassersteinLoss(nn.Module):
    def __init__(self,lambda_pretraining = 1e-5):
        super().__init__()
        self.lambda_pretraining = lambda_pretraining


    def forward(self, mean_out, cov_out, pos_mean_emb, pos_cov_emb):
        '''mean_out = mean_out / torch.max(torch.abs(mean_out))
        cov_out = cov_out / torch.max(torch.abs(cov_out))
        pos_mean_emb = pos_mean_emb / torch.max(torch.abs(pos_mean_emb))
        pos_cov_emb = pos_cov_emb / torch.max(torch.abs(pos_cov_emb))'''
        mean_out = torch.sigmoid(mean_out)
        cov_out = torch.sigmoid(cov_out)
        pos_mean_emb = torch.sigmoid(pos_mean_emb)
        pos_cov_emb = torch.sigmoid(pos_cov_emb)

        pos_logits = wasserstein_distance(mean_out, cov_out, pos_mean_emb, pos_cov_emb)
        pos_logits = pos_logits / torch.max(torch.abs(pos_logits))
        loss = -torch.log(torch.sigmoid(- pos_logits + 1e-24))
        loss = loss/torch.max(torch.abs(loss))
        loss = torch.sum(loss) * self.lambda_pretraining


        return loss


class WassersteinLossFineTuning(nn.Module):
    def __init__(self, lambda_finetuning = 1e-4, lambda_pvn = 1e-4):
        super().__init__()
        self.lambda_finetuning = lambda_finetuning
        self.lambda_pvn = lambda_pvn

    def forward(self, mean_out, cov_out, pos_mean_emb, pos_cov_emb, neg_mean_emb, neg_cov_emb):
        mean_out = torch.sigmoid(mean_out)
        cov_out = torch.sigmoid(cov_out)
        pos_mean_emb = torch.sigmoid(pos_mean_emb)
        pos_cov_emb = torch.sigmoid(pos_cov_emb)
        neg_mean_emb = torch.sigmoid(neg_mean_emb)
        neg_cov_emb = torch.sigmoid(neg_cov_emb)

        pos_logits = wasserstein_distance(mean_out, cov_out, pos_mean_emb, pos_cov_emb)
        neg_logits = wasserstein_distance(mean_out, cov_out, neg_mean_emb, neg_cov_emb)
        pos_vs_neg = wasserstein_distance(pos_mean_emb, pos_cov_emb, neg_mean_emb, neg_cov_emb)

        pos_logits = pos_logits / torch.max(torch.abs(pos_logits))
        neg_logits = neg_logits / torch.max(torch.abs(neg_logits))
        pos_vs_neg = pos_vs_neg / torch.max(torch.abs(pos_vs_neg))

        loss = -torch.log(torch.sigmoid(neg_logits - pos_logits + 1e-24))
        #loss = loss / torch.max(torch.abs(loss)) * 1e-3
        #loss = torch.sigmoid(loss)*0.01
        loss = loss / torch.max(torch.abs(loss)) * self.lambda_finetuning
        loss = torch.sum(loss)

        pvn_loss = torch.clamp(pos_logits - pos_vs_neg, 0)
        #pvn_loss = pvn_loss / torch.max(torch.abs(pvn_loss)) * 1e-3
        #pvn_loss = torch.sigmoid(pvn_loss)*0.01 trial 49
        pvn_loss = pvn_loss / torch.max(torch.abs(pvn_loss)) * self.lambda_pvn
        pvn_loss = torch.sum(pvn_loss)




        return loss + pvn_loss


def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)

    return ret