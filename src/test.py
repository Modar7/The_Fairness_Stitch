import torch
from torch.utils.data import TensorDataset
from src.compute_abroca import *
from .utils import train_test_classifier
from .utils import get_pred, get_pred_Stitched_Model, print_acc_auc_stats
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
def Testing_fun(Trained_TFS, Finetuned_FDR, model, TEST_BS, 
                            trainloader, valloader, testloader, 
                            prepare_data, get_pred_Stitched_Model, 
                            ):
    

    ########################################
    #   Prepare the dataset and save them  #
    ########################################
    x_train, y_train, a_train = prepare_data(trainloader, model, device)
    x_test, y_test, a_test = prepare_data(testloader, model, device)
    x_valid, y_valid, a_valid = prepare_data(valloader, model, device)
    x_finetune, y_finetune, a_finetune = x_valid, y_valid, a_valid
    


    x_train, y_train, a_train = torch.load('datasets/x_train_CelebA_dataset-files'), torch.load('datasets/y_train_CelebA_dataset-files'), torch.load('datasets/a_train_CelebA_dataset-files')
    x_test, y_test, a_test = torch.load('datasets/x_test_CelebA_dataset-files'), torch.load('datasets/y_test_CelebA_dataset-files'), torch.load('datasets/a_test_CelebA_dataset-files')
    x_valid, y_valid, a_valid = torch.load('datasets/x_valid_CelebA_dataset-files'), torch.load('datasets/y_valid_CelebA_dataset-files'), torch.load('datasets/a_valid_CelebA_dataset-files')
  

    Trained_TFS.eval()

    ############
    # Test TFS #
    ############
    out_train, pred_train = get_pred_Stitched_Model(x_train.to(device), Trained_TFS, TEST_BS)
    out_finetune, pred_finetune = get_pred_Stitched_Model(x_finetune.to(device), Trained_TFS, TEST_BS)
    out_test, pred_test = get_pred_Stitched_Model(x_test.to(device), Trained_TFS, TEST_BS)


    sensitive_attrs = ['gender']
    y_train, y_finetune, y_test = y_train.numpy(), y_finetune.numpy(), y_test.numpy()
    a_train, a_finetune, a_test = a_train.numpy(), a_finetune.numpy(), a_test.numpy()
    a_train, a_finetune, a_test = {'gender': a_train}, {'gender': a_finetune}, {'gender': a_test}
    
    train_test_classifier(out_train, out_finetune, out_test, pred_train, pred_finetune, pred_test, y_train, a_train,
                    y_finetune, a_finetune, y_test, a_test,
                    sensitive_attrs)

    print("\n-----------------------------------------------------------------------------------\n")

    out_test22, pred_test22 = out_test, pred_test
    df_pred_proba = pd.DataFrame(out_test22[:, 0].astype(np.float32), columns = ['pred_proba'])
    df_true_label = pd.DataFrame(y_test, columns = ['true_label'])
    df_protected_attribute = pd.DataFrame(a_test['gender'], columns = ['sex'])
    result_x = pd.concat([df_true_label, df_pred_proba, df_protected_attribute], axis=1)
    protected_attribute = 'sex'
    majority_group_name = "Male"
    minority_group_name = "Female"   
    feature = result_x.keys().tolist()    
    filename = "compas.recid.abroca_SL_LS.pdf"
    #Compute Abroca
    slice = compute_abroca(result_x, pred_col = 'pred_proba' , label_col = 'true_label', protected_attr_col = protected_attribute,
                           majority_protected_attr_val = 1, n_grid = 10000,
                           plot_slices = True, majority_group_name=majority_group_name ,minority_group_name=minority_group_name,file_name = filename)
    print("ABROCA:",slice)

    ############
    # Test FDR #
    ############
    Finetuned_FDR.eval()
    out_train, pred_train = get_pred(x_train, Finetuned_FDR, TEST_BS)  
    out_finetune, pred_finetune = get_pred(x_finetune, Finetuned_FDR, TEST_BS) 
    out_test, pred_test = get_pred(x_test, Finetuned_FDR, TEST_BS) 
    sensitive_attrs = ['gender']
    y_train, y_finetune, y_test = y_train.numpy(), y_finetune.numpy(), y_test.numpy()
    a_train, a_finetune, a_test = a_train.numpy(), a_finetune.numpy(), a_test.numpy()
    a_train, a_finetune, a_test = {'gender': a_train}, {'gender': a_finetune}, {'gender': a_test}

    train_test_classifier(out_train, out_finetune, out_test, pred_train, pred_finetune, pred_test, y_train, a_train,
                    y_finetune, a_finetune, y_test, a_test,
                    sensitive_attrs)
    
    df_true_label = pd.DataFrame(y_test, columns = ['true_label'])
    df_pred_proba = pd.DataFrame(out_test, columns = ['pred_proba'])
    out_test22, pred_test22 = out_test, pred_test
    df_pred_proba = pd.DataFrame(out_test22[:, 0].astype(np.float32), columns = ['pred_proba'])
    df_true_label = pd.DataFrame(y_test, columns = ['true_label'])
    df_protected_attribute = pd.DataFrame(a_test['gender'], columns = ['sex'])
    result_x = pd.concat([df_true_label, df_pred_proba, df_protected_attribute], axis=1)   
    feature = result_x.keys().tolist()    
    filename = "compas.recid.abroca_No_SL.pdf"
    #Compute Abroca
    slice = compute_abroca(result_x, pred_col = 'pred_proba' , label_col = 'true_label', protected_attr_col = protected_attribute,
                       majority_protected_attr_val = 1, n_grid = 10000,
                       plot_slices = True, majority_group_name=majority_group_name ,minority_group_name=minority_group_name,file_name = filename)
    print("ABROCA:",slice)
