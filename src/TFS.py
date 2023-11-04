"""
This Python script contains some code adapted from the following GitHub repository.

Original source: [https://github.com/yuzhenmao/Fairness-Finetuning/tree/main]

"""


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from src.metrics import *
from .fairness_constraints import *
from .utils import get_pred, get_pred_Stitched_Model, print_acc_auc_stats


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def Training_Stitched_Model(Stitched_Model, model, optimizer, criterion, 
                            epoch, TEST_BS, 
                            trainloader, valloader, testloader, 
                            prepare_data, get_pred_Stitched_Model, 
                            print_acc_auc_stats, args):
   
    losses = []
    model.eval()
    ########################################
    #   Prepare the dataset and save them  #
    ########################################
    x_train, y_train, a_train = prepare_data(trainloader, model, device)
    x_test, y_test, a_test = prepare_data(testloader, model, device)
    x_valid, y_valid, a_valid = prepare_data(valloader, model, device)
    x_finetune, y_finetune, a_finetune = x_valid, y_valid, a_valid


    torch.save(x_train,'datasets/x_train_CelebA_dataset-files'), torch.save(y_train,'datasets/y_train_CelebA_dataset-files'), torch.save(a_train,'datasets/a_train_CelebA_dataset-files')
    torch.save(x_test,'datasets/x_test_CelebA_dataset-files'), torch.save(y_test,'datasets/y_test_CelebA_dataset-files'), torch.save(a_test,'datasets/a_test_CelebA_dataset-files')
    torch.save(x_valid,'datasets/x_valid_CelebA_dataset-files'), torch.save(y_valid, 'datasets/y_valid_CelebA_dataset-files'), torch.save(a_valid,'datasets/a_valid_CelebA_dataset-files')

    ################################################################
    #         Prepare the balanced dataset for training TFS        #
    ################################################################
    #              Sample a balanced dataset and save it           #
    ################################################################
    ################################################################
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
    #print(min_g)
    temp_g = torch.cat([g[:min_g] for g in g_idx])
    x_finetune = X[temp_g]
    y_finetune = hair[temp_g]
    a_finetune = gender[temp_g]
    

    x_balanced, y_balanced, a_balanced = x_finetune, y_finetune, a_finetune
    torch.save(x_finetune,'datasets/x_balanced_CelebA_dataset-files'), torch.save(y_finetune, 'datasets/y_balanced_CelebA_dataset-files'), torch.save(a_finetune,'datasets/a_balanced_CelebA_dataset-files')
    ################################################################
    ################################################################
    
    balanced_dataset = TensorDataset(x_balanced, y_balanced, a_balanced)
    if args.batch_size < 0:
        batch_size = y_balanced.shape[0]
    else:
        batch_size = args.batch_size
    balancedloader = torch.utils.data.DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
    print('-----------------------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------------------')
    print('------------------------------------------Training TFS-----------------------------------------')
    ################################################################
    #                         Training TFS                         #
    ################################################################
    for epoch in range(1, args.ft_epoch + 1):
        epoch_loss = 0.0
        epoch_loss_fairness = 0.0
        epoch_acc = 0.0
        Stitched_Model.train()
        for batch_idx, (x, y, a) in enumerate(balancedloader):
            x, y, a = x.to(device), y.to(device), a.to(device)
            optimizer.zero_grad()
            outputs = Stitched_Model(x) 
            log_softmax, softmax = F.log_softmax(outputs, dim=1), F.softmax(outputs, dim=1)
            if args.constraint == 'MMF':
                loss = mmf_constraint(criterion, log_softmax, y, a)
            else:
                if args.constraint == 'EO':
                    fpr, fnr = eo_constraint(softmax[:, 1], y, a)
                    loss_fairness = fpr + fnr
                elif args.constraint == 'AE':
                    loss_fairness = ae_constraint(criterion, log_softmax, y, a)
                epoch_loss_fairness += loss_fairness.item()
                loss_1 = criterion(log_softmax, y)
                loss = loss_1 + args.alpha * loss_fairness
            #else:
              #  loss = criterion(log_softmax, y)
    
            epoch_loss += loss.item()
    
            loss.backward(retain_graph=True)
            optimizer.step()
    
            _, preds = torch.max(outputs.data, 1)
            epoch_acc += torch.sum(preds == y).item()

    
        epoch_loss /= len(balancedloader)
        epoch_loss_fairness /= len(balancedloader)
        epoch_acc /= len(balanced_dataset)
        losses.append(epoch_loss)
        print('Training TFS: Epoch %d/%d   Loss: %.4f' % (
            epoch, args.ft_epoch, epoch_loss))
        

        #print('--------------------------------------------------------')
        print('Validating TFS: Epoch %d/%d    ' % (epoch, args.ft_epoch))
        ################################################################
        #                         Validating TFS                       #
        ################################################################
        x_train, y_train, a_train = torch.load('datasets/x_train_CelebA_dataset-files'), torch.load('datasets/y_train_CelebA_dataset-files'), torch.load('datasets/a_train_CelebA_dataset-files')
        x_test, y_test, a_test = torch.load('datasets/x_test_CelebA_dataset-files'), torch.load('datasets/y_test_CelebA_dataset-files'), torch.load('datasets/a_test_CelebA_dataset-files')
        x_valid, y_valid, a_valid = torch.load('datasets/x_valid_CelebA_dataset-files'), torch.load('datasets/y_valid_CelebA_dataset-files'), torch.load('datasets/a_valid_CelebA_dataset-files')
        x_balanced, y_balanced, a_balanced = torch.load('datasets/x_balanced_CelebA_dataset-files'), torch.load('datasets/y_balanced_CelebA_dataset-files'), torch.load('datasets/a_balanced_CelebA_dataset-files')
        #x_finetune, y_finetune, a_finetune = x_balanced, y_balanced, a_balanced 

        Stitched_Model.eval()
        out_train, pred_train = get_pred_Stitched_Model(x_train.to(device), Stitched_Model, TEST_BS)
        out_valid, pred_valid = get_pred_Stitched_Model(x_valid.to(device), Stitched_Model, TEST_BS)
        out_balanced, pred_balanced = get_pred_Stitched_Model(x_balanced.to(device), Stitched_Model, TEST_BS)
        out_test, pred_test = get_pred_Stitched_Model(x_test.to(device), Stitched_Model, TEST_BS)
        sensitive_attrs = ['gender']
        y_train, y_valid, y_balanced, y_test = y_train.numpy(), y_valid.numpy(), y_balanced.numpy(), y_test.numpy()
        a_train, a_valid, a_balanced, a_test = a_train.numpy(), a_valid.numpy(), a_balanced.numpy(), a_test.numpy()
        a_train, a_valid, a_balanced, a_test = {'gender': a_train}, {'gender': a_valid}, {'gender': a_balanced}, {'gender': a_test}
        
        balanced_BACC = balanced_accuracy_score(y_balanced, pred_balanced)
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
            state = {
            'epoch': epoch,
            'state_dict': Stitched_Model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
            savepath=f'checkpoint_trained_FDR_model_Epochs{epoch}_CelebA_valid.t7'
            torch.save(state,savepath)
            args.valid_output_xlsx_info = args.valid_output_xlsx_info.append({'valid eod': valid_eod, 'valid BACC': valid_BACC, 'valid auc': valid_auc, 'epoch': epoch, 'savepath': savepath}, ignore_index=True)
        print('-----------------------------------------------------------------------------------------------')
        print('-----------------------------------------------------------------------------------------------')
    args.valid_output_xlsx_info.to_excel('output_TFS_CelebA.xlsx', index=False)
        
    
          
    
    