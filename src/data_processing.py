import torch

def prepare_data(dataloader, model, device):
    x_, y_, a_ = [], [], []
    for batch_idx, (images, labels) in enumerate(dataloader):
        gender = labels[:, 20]
        a_.append(gender)
        hair = labels[:, 9]
        y_.append(hair)
        X = model.get_features(images.to(device)).detach().cpu()
        x_.append(X)

    return torch.cat(x_), torch.cat(y_), torch.cat(a_)