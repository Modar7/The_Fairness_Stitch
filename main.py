"""
This Python script contains some code adapted from the following GitHub repository.

Original source: [https://github.com/yuzhenmao/Fairness-Finetuning/tree/main]

"""

import sys
import easydict
import numpy as np
from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score,  accuracy_score, roc_auc_score
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset


from src.models import MyResNet, Stitched_Model
from src.data_processing import prepare_data
from src import train_per_epoch, valid_per_epoch, Finetune
from src import Training_Stitched_Model
from src import get_pred, get_pred_Stitched_Model, print_acc_auc_stats


args = easydict.EasyDict({
    "method": 'M2',
    "ft_epoch": 1000,
    "ft_lr": 1e-2,
    "alpha": 20,
    "constrain": 'EO',
    "seed": 202212,
    "data_path": 'datasets',
    "batch_size": 1,
    "constraint": 'EO',
    "checkpoint": 'checkpoint_trained_model_Step_1.t7',
    "valid_output_xlsx_info": pd.DataFrame(columns=['valid eod', 'valid BACC', 'valid auc', 'epoch', 'savepath']),
})

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

data_root = args.data_path
train_dataset = datasets.CelebA(data_root, split="train", target_type=["attr"], transform=transform)
valid_dataset = datasets.CelebA(data_root, split="valid", target_type=["attr"], transform=transform)
test_dataset =  datasets.CelebA(data_root, split="test", target_type=["attr"], transform=transform)

train_subset = train_dataset 
valid_subset = valid_dataset
test_subset = test_dataset 
TRAIN_BS = 1024
TEST_BS = 2048
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=TRAIN_BS,
                                          shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valid_subset, batch_size=TEST_BS,
                                        shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(test_subset, batch_size=TEST_BS,
                                         shuffle=False, num_workers=4)



#valid_output_xlsx_info = pd.DataFrame(columns=['valid eod', 'valid BACC', 'valid auc', 'epoch', 'savepath'])


def main():
    model = MyResNet(num_classes=2, pretrain=False)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    if args.checkpoint is not None:
        print('Recovering from %s ...' % (args.checkpoint))
        checkpoint = torch.load(args.checkpoint)  #, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
    else:
        ##################################################################################################
        #     Step 1 in FDR and TFS: Training the Resnet18 model to get the trained model for Step 2     #
        ##################################################################################################
        NUM_EPOCHS = 1000
        losses = []
        trigger_times = 0
        best_loss = 1e9
        best_model = None
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, last_epoch=-1)
        for epoch in range(NUM_EPOCHS):
            train_per_epoch(model, optimizer, criterion, epoch + 1, NUM_EPOCHS, trainloader, train_subset)
            epoch_loss = valid_per_epoch(model, epoch + 1, NUM_EPOCHS, criterion, valloader, valid_subset)
            losses.append(epoch_loss)
            if epoch_loss < best_loss and epoch > 0:
                best_model = deepcopy(model)
                best_loss = epoch_loss
            # Early Stop
            if (epoch > 20) and (losses[-1] >= losses[-2]):
                trigger_times += 1
                if trigger_times > 2:
                    break
            else:
                trigger_times = 0

            scheduler.step()
        ###################################################
        # Save the trained model in Step 1 in FDR and TFS #
        ###################################################
        state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }

        savepath='checkpoint_trained_model_Step_1.t7'
        torch.save(state,savepath)   

    ###################################################################################################
    #                                     FDR: Finetune and test                                      #                                  
    ###################################################################################################
    Finetune(model, criterion, trainloader, valloader, testloader, prepare_data, TEST_BS, args)
    ###################################################################################################
    # TFS: Train the stitched model and save different ckeckpoints that satisfy good enough trade-off #
    ###################################################################################################
    TFS_model = Stitched_Model(model)
    for name, param in TFS_model.named_parameters():
        if param.requires_grad and 'last_layer' in name:
            param.requires_grad = False
    non_frozen_parameters = [p for p in TFS_model.parameters() if p.requires_grad]
    #optimizer = optim.SGD(non_frozen_parameters, lr=0.1)
    optimizer = optim.SGD(non_frozen_parameters, lr=args.ft_lr, momentum=0.9, weight_decay=5e-4)
    Training_Stitched_Model(TFS_model, model, optimizer, criterion, 
                            epoch, TEST_BS, 
                            trainloader, valloader, testloader, 
                            prepare_data, get_pred_Stitched_Model,
                            print_acc_auc_stats, args)
         

if __name__ == '__main__':
    main()

    