import numpy as np
import torch
from copy import deepcopy


def accuracy_equality_difference(y, pred, sensitive_features):
    misclassification_rate_p = sum(y[sensitive_features == 1] != pred[sensitive_features == 1]) / sum(
        sensitive_features == 1)
    misclassification_rate_n = sum(y[sensitive_features == 0] != pred[sensitive_features == 0]) / sum(
        sensitive_features == 0)
    return abs(misclassification_rate_p - misclassification_rate_n)


def max_min_fairness(y, pred, sensitive_features):
    y_p_a = y + sensitive_features
    y_m_a = y - sensitive_features
    classification_rate_1 = sum(y[y_p_a == 2] == pred[y_p_a == 2]) / sum(y_p_a == 2)
    classification_rate_2 = sum(y[y_p_a == 0] == pred[y_p_a == 0]) / sum(y_p_a == 0)
    classification_rate_3 = sum(y[y_m_a == 1] == pred[y_m_a == 1]) / sum(y_m_a == 1)
    classification_rate_4 = sum(y[y_m_a == -1] == pred[y_m_a == -1]) / sum(y_m_a == -1)
    return min(min(classification_rate_1, classification_rate_2), min(classification_rate_3, classification_rate_4))

