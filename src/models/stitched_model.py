import torch
import torch.nn as nn


#####################
# The Stitched Model
#####################

# Define the modified neural network*
class Stitched_Model(nn.Module):
    def __init__(self, model_x):
        super(Stitched_Model, self).__init__()
        self.stitching_layer = nn.Linear(512, 512)
        self.last_layer = model_x.resnet18.fc[0]
        
    def forward(self, x):
        x = self.stitching_layer(x) # Fair Head
        x = self.last_layer(x)
        return x
    


