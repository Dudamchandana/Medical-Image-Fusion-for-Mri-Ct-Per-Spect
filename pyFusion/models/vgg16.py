import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

class VGG16(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(VGG16, self).__init__()

        # Convolutional Levels the classifier is not used
        features = list(vgg16(pretrained=True).features)
        
        if device == "cuda":
            self.features = nn.ModuleList(features).cuda().eval()
        else:
            self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        feature_maps = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            #print('level index: ' + str(idx) )
            if idx == 1:
                #Choose the map extracted at the idx-th level
                feature_maps.append(x)
        return feature_maps
