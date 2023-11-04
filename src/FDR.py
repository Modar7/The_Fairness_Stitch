"""
This Python script contains some code adapted from the following GitHub repository.

Original source: [https://github.com/yuzhenmao/Fairness-Finetuning/tree/main]

"""


import os, sys
import numpy as np

import torch
from torch.utils.data import TensorDataset
import torchvision.models as models
import torch.optim as optim
from torchvision.transforms import Normalize
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
import argparse
import pandas as pd

from src.metrics import *
from .fairness_constraints import *
from .utils import get_pred, train_test_classifier

from fairlearn.reductions import GridSearch, EqualizedOdds, ExponentiatedGradient
from fairlearn.metrics import (
    MetricFrame, equalized_odds_difference, equalized_odds_ratio,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from .utils import print_acc_auc_stats

#from utils import *
#from models import MyResNet, ConvNet

#from .data_processing import prepare_data
#from src.data_processing import prepare_data
#from models import MyResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_per_epoch(model, optimizer, criterion, epoch, num_epochs, trainloader, train_dataset):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, (images, labels) in enumerate(trainloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # move to GPU
        images, labels = images.to(device), labels[:, 9].to(device)

        # forward
        outputs = model.forward(images)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels).item()

    epoch_loss /= len(trainloader)
    epoch_acc /= len(train_dataset)

    print('TRAINING Epoch %d/%d Loss %.4f Accuracy %.4f' % (epoch, num_epochs, epoch_loss, epoch_acc))


def valid_per_epoch(model, epoch, num_epochs, criterion, valloader, valid_dataset):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, (images, labels) in enumerate(valloader):
        # move to GPU
        images, labels = images.to(device), labels[:, 9].to(device)

        # forward
        outputs = model.forward(images)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)

        epoch_loss += loss.item()
        epoch_acc += torch.sum(preds == labels).item()

    epoch_loss /= len(valloader)
    epoch_acc /= len(valid_dataset)

    print('VALID Epoch %d/%d Loss %.4f Accuracy %.4f' % (epoch, num_epochs, epoch_loss, epoch_acc))

    return epoch_loss



def Finetune(model, criterion, trainloader, valloader, testloader, prepare_data, TEST_BS, args):
    model.eval()

    best_clf = None
    method = args.method

    ###################################################
    # (M2): Finetune on balanced-sampled + constraint
    ###################################################

    ################
    # Prepare the dataset
    ################
    x_train, y_train, a_train = prepare_data(trainloader, model, device)
    x_test, y_test, a_test = prepare_data(testloader, model, device)
    x_valid, y_valid, a_valid = prepare_data(valloader, model, device)
    x_finetune, y_finetune, a_finetune = prepare_data(valloader, model, device)

    torch.save(x_train,'datasets/x_train_CelebA_dataset-files'), torch.save(y_train,'y_train_CelebA_dataset-files'), torch.save(a_train,'a_train_CelebA_dataset-files')
    torch.save(x_test,'x_test_CelebA_dataset-files'), torch.save(y_test,'y_test_CelebA_dataset-files'), torch.save(a_test,'a_test_CelebA_dataset-files')
    torch.save(x_finetune,'x_valid_CelebA_dataset-files'), torch.save(y_finetune,'y_valid_CelebA_dataset-files'), torch.save(a_finetune,'a_valid_CelebA_dataset-files')

    if method == 'B1':
        x_finetune = x_train
        y_finetune = y_train
        a_finetune = a_train

    elif method == 'B2' or method == 'M1':
        pass

    elif method == 'B3' or method == 'M2':  # Sample a balanced dataset
        X = torch.cat([x_train, x_finetune])
        hair = torch.cat([y_train, y_finetune])
        gender = torch.cat([a_train, a_finetune])
        g_idx = []
        g_idx.append(torch.where((gender + hair) == 2)[0])  # (1, 1)
        g_idx.append(torch.where((gender + hair) == 0)[0])  # (0, 0)
        g_idx.append(torch.where((gender - hair) == 1)[0])  # (1, 0)
        g_idx.append(torch.where((gender - hair) == -1)[0])  # (0, 1)
        for i, g in enumerate(g_idx):
            idx = torch.randperm(g.shape[0])
            g_idx[i] = g[idx]
        min_g = min([len(g) for g in g_idx])
        print(min_g)
        temp_g = torch.cat([g[:min_g] for g in g_idx])
        x_finetune = X[temp_g]
        y_finetune = hair[temp_g]
        a_finetune = gender[temp_g]

    #############
    # Fine-tune #
    #############
    model.train()
    model.set_grad(False)
    model.append_last_layer()
    model = model.to(device)
    optimizer = optim.SGD(model.out_fc.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epoch)
    finetune_dataset = TensorDataset(x_finetune, y_finetune, a_finetune)
    # For B3 and M2, considering balance, only tried full batch
    if args.batch_size < 0:
        batch_size = y_finetune.shape[0]
    else:
        batch_size = args.batch_size
    print(batch_size)
    finetuneloader = torch.utils.data.DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)
    print(len(finetune_dataset))

    weights = (max(torch.bincount(y_finetune))/torch.bincount(y_finetune))
    class_weights = torch.FloatTensor(weights).to(device)
    print(torch.bincount(y_finetune))
    print(class_weights)

    losses = []
    trigger_times = 0
    best_loss = 1e9
    for epoch in range(1, args.ft_epoch + 1):
        epoch_loss = 0.0
        epoch_loss_fairness = 0.0
        epoch_acc = 0.0
        for batch_idx, (x, y, a) in enumerate(finetuneloader):
            x, y, a = x.to(device), y.to(device), a.to(device)
            optimizer.zero_grad()
            outputs = model.out_fc(x)
            log_softmax, softmax = F.log_softmax(outputs, dim=1), F.softmax(outputs, dim=1)
            if method == 'M2':  # Use the fairness constraint
                if args.constraint == 'MMF':
                    loss = mmf_constraint(criterion, log_softmax, y, a)
                else:
                    if args.constraint == 'EO':
                        fpr, fnr = eo_constraint(softmax[:, 1], y, a)
                        loss_fairness = fpr + fnr
                    elif args.constraint == 'AE':
                        loss_fairness = ae_constraint(criterion, log_softmax, y, a)
                    epoch_loss_fairness += loss_fairness.item()
                    loss_1 = nn.NLLLoss(weight=class_weights)(log_softmax, y)
                    loss = loss_1 + args.alpha * loss_fairness
            else:
                loss = nn.NLLLoss(weight=class_weights)(log_softmax, y)

            epoch_loss += loss.item()

            loss.backward(retain_graph=True)
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)
            epoch_acc += torch.sum(preds == y).item()

        # scheduler.step()

        epoch_loss /= len(finetuneloader)
        epoch_loss_fairness /= len(finetuneloader)
        epoch_acc /= len(finetune_dataset)
        losses.append(epoch_loss)
        print('FINETUNE Epoch %d/%d   Loss_1: %.4f   Loss_2: %.4f   Accuracy: %.4f' % (
            epoch, args.ft_epoch, epoch_loss, epoch_loss_fairness, epoch_acc))

    model.eval()
    out_train, pred_train = get_pred(x_train, model, TEST_BS)
    out_finetune, pred_finetune = get_pred(x_finetune, model, TEST_BS)
    out_valid, pred_valid = get_pred(x_valid, model, TEST_BS)
    out_test, pred_test = get_pred(x_test, model, TEST_BS)
    sensitive_attrs = ['gender']
    y_train, y_finetune, y_valid, y_test = y_train.numpy(), y_finetune.numpy(), y_valid.numpy(), y_test.numpy()
    a_train, a_finetune, a_valid, a_test = a_train.numpy(), a_finetune.numpy(), a_valid.numpy(), a_test.numpy()
    a_train, a_finetune, a_valid, a_test = {'gender': a_train}, {'gender': a_finetune}, {'gender': a_valid}, {'gender': a_test}

    train_test_classifier(out_train, out_finetune, out_test, pred_train, pred_finetune, pred_test, y_train, a_train,
                    y_finetune, a_finetune, y_test, a_test,
                    sensitive_attrs)
    print("\n-----------------------------------------------------------------------------------\n")
    state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
    }
    savepath=f'FDR_model_CelebA.t7'
    torch.save(state,savepath)


    #balanced_BACC = balanced_accuracy_score(y_balanced, pred_balanced)
    valid_BACC = balanced_accuracy_score(y_valid, pred_valid)
    valid_acc = print_acc_auc_stats(out_train, out_valid, out_test, pred_train, pred_valid, pred_test, y_train, a_train,
                            y_valid, a_valid, y_test, a_test,
                            sensitive_attrs)[0]
    valid_auc = print_acc_auc_stats(out_train, out_valid, out_test, pred_train, pred_valid, pred_test, y_train, a_train,
                             y_valid, a_valid, y_test, a_test,
                            sensitive_attrs)[1]   
    valid_eod = equalized_odds_difference(y_valid, pred_valid, sensitive_features=a_valid)
    test_aed = accuracy_equality_difference(y_test, pred_test, sensitive_features=a_test['gender'])
    test_mmf = max_min_fairness(y_test, pred_test, sensitive_features=a_test['gender'])
    #print('-----------------------------------------------------------------------------------------------')
    #print('-----------------------------------------------------------------------------------------------')
    if valid_BACC > 0.86 and valid_auc > 0.93 and valid_eod < 0.12:
        #print('There is a good performance-fairness trade-off')
        #print("Valid AUC: %0.3f" % (valid_auc))
        #print(f'Valid eod: {valid_eod}')
        #print(f'Valid BACC: {valid_BACC}')
        #print('epoch:', epoch)
        state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
        savepath=f'checkpoint_trained_FDR_model_Epochs{epoch}_CelebA_valid.t7'
        torch.save(state,savepath)
        args.valid_output_xlsx_info = args.valid_output_xlsx_info.append({'valid eod': valid_eod, 'valid BACC': valid_BACC, 'valid auc': valid_auc, 'epoch': epoch, 'savepath': savepath}, ignore_index=True)
    #else:
      #  print('There is no good performance-fairness trade-off')
    print('-----------------------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------------------')
    args.valid_output_xlsx_info.to_excel('output_FDR_CelebA.xlsx', index=False)
