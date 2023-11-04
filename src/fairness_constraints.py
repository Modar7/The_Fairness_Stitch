"""
This Python script contains some code adapted from a GitHub repository.

Original source: [https://github.com/yuzhenmao/Fairness-Finetuning/tree/main]

"""

import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, \
    average_precision_score
from fairlearn.metrics import (
    MetricFrame, equalized_odds_difference, equalized_odds_ratio,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)



def eo_constraint(p, y, a):
    fpr = torch.abs(torch.sum(p * (1 - y) * a) / (torch.sum(a) + 1e-5) - torch.sum(p * (1 - y) * (1 - a)) / (
                torch.sum(1 - a) + 1e-5))
    fnr = torch.abs(torch.sum((1 - p) * y * a) / (torch.sum(a) + 1e-5) - torch.sum((1 - p) * y * (1 - a)) / (
                torch.sum(1 - a) + 1e-5))
    return fpr, fnr


def ae_constraint(criterion, log_softmax, y, a):
    loss_p = criterion(log_softmax[a == 1], y[a == 1])
    loss_n = criterion(log_softmax[a == 0], y[a == 0])
    return torch.abs(loss_p - loss_n)


def mmf_constraint(criterion, log_softmax, y, a):
    # loss_p = criterion(log_softmax[a == 1], y[a == 1])
    # loss_n = criterion(log_softmax[a == 0], y[a == 0])
    # return torch.max(loss_p, loss_n)
    y_p_a = y + a
    y_m_a = y - a
    if len(y[y_p_a == 2]) > 0:
        loss_1 = criterion(log_softmax[y_p_a == 2], y[y_p_a == 2])  # (1, 1)
    else:
        loss_1 = torch.tensor(0.0).cuda()
    if len(y[y_p_a == 0]) > 0:
        loss_2 = criterion(log_softmax[y_p_a == 0], y[y_p_a == 0])  # (0, 0)
    else:
        loss_2 = torch.tensor(0.0).cuda()
    if len(y[y_m_a == 1]) > 0:
        loss_3 = criterion(log_softmax[y_m_a == 1], y[y_m_a == 1])  # (1, 0)
    else:
        loss_3 = torch.tensor(0.0).cuda()
    if len(y[y_m_a == -1]) > 0:
        loss_4 = criterion(log_softmax[y_m_a == -1], y[y_m_a == -1])  # (0, 1)
    else:
        loss_4 = torch.tensor(0.0).cuda()
    return torch.max(torch.max(loss_1, loss_2), torch.max(loss_3, loss_4))