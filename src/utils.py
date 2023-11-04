

import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import balanced_accuracy_score,  accuracy_score, roc_auc_score
from src.metrics import accuracy_equality_difference, max_min_fairness
from fairlearn.metrics import (equalized_odds_difference, false_positive_rate, false_negative_rate)

######
# Test
######

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pred(x, model, TEST_BS):  # Avoid exceeding the memory limit  get_pred(x, model, TEST_BS): 
    dataset = TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=TEST_BS, shuffle=False)
    outs = []
    for x in loader:
        out = F.softmax(model.out_fc(x[0].to(device)), dim=1).cpu().detach().numpy()
        outs.append(out)
    outs = np.concatenate(outs)
    pred = np.argmax(outs, 1)
    return outs[:, 1], pred



def get_pred_Stitched_Model(x, Stitched_Model, TEST_BS):  # Avoid exceeding the memory limit
    dataset = TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=TEST_BS, shuffle=False)
    outs = []
    for x in loader:
        #out = F.softmax(model.resnet18.fc[0](x[0]).to(device), dim=1).cpu().detach().numpy()   # model.out_fc(x[0]).to(device)
        out = F.softmax(Stitched_Model(x[0].to(device)), dim=1).cpu().detach().numpy()
        #out = F.softmax(model.out_fc(x[0]).to(device), dim=1).cpu().detach().numpy()   # model.out_fc(x[0]).to(device)
        outs.append(out)
    outs = np.concatenate(outs)
    pred = np.argmax(outs, 1)
    return outs[:, 1], pred


def print_fpr_fnr_sensitive_features(y_true, y_pred, x_control, sensitive_attrs):
    for s in sensitive_attrs:
        s_attr_vals = x_control[s]
        print("||  s  || FPR. || FNR. ||")
        for s_val in sorted(list(set(s_attr_vals))):
            y_true_local = y_true[s_attr_vals == s_val]
            y_pred_local = y_pred[s_attr_vals == s_val]

            fpr = false_positive_rate(y_true_local, y_pred_local)
            fnr = false_negative_rate(y_true_local, y_pred_local)

            if isinstance(s_val, float):  # print the int value of the sensitive attr val
                s_val = int(s_val)
            print("||  %s  || %0.2f || %0.2f ||" % (s_val, fpr, fnr))


def print_clf_stats(out_train, out_finetune, out_test, pred_train, pred_finetune, pred_test, y_train, a_train,
                    y_finetune, a_finetune, y_test, a_test, sensitive_attrs):
    train_acc, finetune_acc, test_acc = accuracy_score(y_train, pred_train), accuracy_score(y_finetune,
                                                                                            pred_finetune), accuracy_score(
        y_test, pred_test)
    train_auc, finetune_auc, test_auc = roc_auc_score(y_train, out_train), roc_auc_score(y_finetune,
                                                                                          out_finetune), roc_auc_score(
        y_test, out_test)

    for s_attr in sensitive_attrs:
        print("*** Train ***")
        print("Accuracy: %0.3f, AUC: %0.3f" % (train_acc, train_auc))
        print_fpr_fnr_sensitive_features(y_train, pred_train, a_train, sensitive_attrs)

        print("\n")
        print("*** Finetune ***")
        print("Accuracy: %0.3f, AUC: %0.3f" % (finetune_acc, finetune_auc))
        print_fpr_fnr_sensitive_features(y_finetune, pred_finetune, a_finetune, sensitive_attrs)

        print("\n")
        print("*** Test ***")
        print("Accuracy: %0.3f, AUC: %0.3f" % (test_acc, test_auc))
        print_fpr_fnr_sensitive_features(y_test, pred_test, a_test, sensitive_attrs)
        print("\n")
        

def print_acc_auc_stats(out_train, out_valid, out_test, pred_train, pred_valid, pred_test, y_train, a_train,
                            y_valid, a_valid, y_test, a_test, sensitive_attrs):
            train_acc, valid_acc, test_acc = accuracy_score(y_train, pred_train), accuracy_score(y_valid, pred_valid), accuracy_score(
                y_test, pred_test)
            train_auc, valid_auc, test_auc = roc_auc_score(y_train, out_train), roc_auc_score(y_valid, out_valid), roc_auc_score(
                y_test, out_test)
            return valid_acc, valid_auc

def train_test_classifier(out_train, out_finetune, out_test, pred_train, pred_finetune, pred_test, y_train, a_train,
                    y_finetune, a_finetune, y_test, a_test,
                    sensitive_attrs):
    print_clf_stats(out_train, out_finetune, out_test, pred_train, pred_finetune, pred_test, y_train, a_train,
                    y_finetune, a_finetune, y_test, a_test,
                    sensitive_attrs)

    valid_BACC = balanced_accuracy_score(y_finetune, pred_finetune)
    print(f'Finetune weighted accuracy: {valid_BACC}')
    
    test_BACC = balanced_accuracy_score(y_test, pred_test)
    print(f'Test weighted accuracy: {test_BACC}')

    finetune_eod = equalized_odds_difference(y_finetune, pred_finetune, sensitive_features=a_finetune)
    print("\n")
    print(f'Finetune equalized_odds_difference: {finetune_eod}')

    test_eod = equalized_odds_difference(y_test, pred_test, sensitive_features=a_test)
    print("\n")
    print(f'Test equalized_odds_difference: {test_eod}')


    finetune_aed = accuracy_equality_difference(y_finetune, pred_finetune, sensitive_features=a_finetune['gender'])
    print("\n")
    print(f'Finetune accuracy_equality_difference: {finetune_aed}')
    

    test_aed = accuracy_equality_difference(y_test, pred_test, sensitive_features=a_test['gender'])
    print("\n")
    print(f'Test accuracy_equality_difference: {test_aed}')
    

    finetune_mmf = max_min_fairness(y_finetune, pred_finetune, sensitive_features=a_finetune['gender'])
    print("\n")
    print(f'Finetune max_min_fairness: {finetune_mmf}')

    test_mmf = max_min_fairness(y_test, pred_test, sensitive_features=a_test['gender'])
    print("\n")
    print(f'Test max_min_fairness: {test_mmf}')


    