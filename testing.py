import torch

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data import TensorDataset
from torchvision import transforms


import easydict

from src.compute_abroca import *
from src.test import Testing_fun
from src.models import MyResNet, Stitched_Model
from src.data_processing import prepare_data
from src import get_pred_Stitched_Model




args = easydict.EasyDict({
    "method": 'M2',
    "ft_epoch": 1000,
    "ft_lr": 1e-2,
    "alpha": 20,
    "constraint": 'EO',
    "seed": 202212,
    "data_path": 'datasets',
    "batch_size": 1,
    "constraint": 'EO',
    "checkpoint": 'checkpoint_trained_model_Step_1.t7',
    "valid_output_xlsx_info": pd.DataFrame(columns=['valid eod', 'valid BACC', 'valid auc', 'epoch', 'savepath']),
    "checkpoint_TFS_model": 'checkpoint_trained_FDR_model_Epochsi_CelebA_valid.t7', # The TFS model should be selected from the file output_TFS_CelebA.xlsx after training it.
    "checkpoint_FDR_model": 'checkpoint_trained_FDR_model_Epochsj_CelebA_valid.t7', # The FDR model should be selected from the file output_FDR_CelebA.xlsx after training it.
})

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

data_root = args.data_path
train_dataset = datasets.CelebA(data_root, split="train", target_type=["attr"], transform=transform)
valid_dataset = datasets.CelebA(data_root, split="valid", target_type=["attr"], transform=transform)
test_dataset =  datasets.CelebA(data_root, split="test", target_type=["attr"], transform=transform)

train_subset = Subset(train_dataset, np.arange(1, 26))
valid_subset = Subset(valid_dataset, np.arange(1, 26))
test_subset = Subset(test_dataset, np.arange(1, 26))
TRAIN_BS = 1024
TEST_BS = 2048
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=TRAIN_BS,
                                          shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valid_subset, batch_size=TEST_BS,
                                        shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(test_subset, batch_size=TEST_BS,
                                         shuffle=False, num_workers=4)



x_train, y_train, a_train = torch.load('datasets/x_train_CelebA_dataset-files'), torch.load('datasets/y_train_CelebA_dataset-files'), torch.load('datasets/a_train_CelebA_dataset-files')
x_test, y_test, a_test = torch.load('datasets/x_test_CelebA_dataset-files'), torch.load('datasets/y_test_CelebA_dataset-files'), torch.load('datasets/a_test_CelebA_dataset-files')
x_valid, y_valid, a_valid = torch.load('datasets/x_valid_CelebA_dataset-files'), torch.load('datasets/y_valid_CelebA_dataset-files'), torch.load('datasets/a_valid_CelebA_dataset-files')
x_balanced, y_balanced, a_balanced = torch.load('datasets/x_balanced_CelebA_dataset-files'), torch.load('datasets/y_balanced_CelebA_dataset-files'), torch.load('datasets/a_balanced_CelebA_dataset-files')
#x_finetune, y_finetune, a_finetune = x_balanced, y_balanced, a_balanced 


# load the trained MyResNet model #
model = MyResNet(num_classes=2, pretrain=False)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
checkpoint = torch.load(args.checkpoint)  
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']

# load TFS #
TFS_model = Stitched_Model(model)
for name, param in TFS_model.named_parameters():
        if param.requires_grad and 'last_layer' in name:
            param.requires_grad = False
non_frozen_parameters = [p for p in TFS_model.parameters() if p.requires_grad]
optimizer = optim.SGD(non_frozen_parameters, lr=args.ft_lr, momentum=0.9, weight_decay=5e-4)
checkpoint = torch.load(args.checkpoint_TFS_model)  
TFS_model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']


# load FDR #
model.train()
model.set_grad(False)
model.append_last_layer()
FDR_model = model.to(device)
optimizer = optim.SGD(FDR_model.out_fc.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=5e-4)
checkpoint = torch.load(args.checkpoint_FDR_model)
FDR_model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']



###################################################################################################
# TFS: Tresting TFS and FDR  #
###################################################################################################
Testing_fun(TFS_model, FDR_model, model, TEST_BS, 
                            trainloader, valloader, testloader, 
                            prepare_data, get_pred_Stitched_Model, 
                            )



