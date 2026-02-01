import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models

 # Artem Bortezko kaggle architecture
def CNN_backbone():
    relu = nn.ReLU()
    conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7))
    conv2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(7,7))
    conv3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(5,5))
    conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5))
    conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3))
    conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))
    pool = nn.MaxPool2d(kernel_size=(2,2))
    conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))
    bn1 = nn.BatchNorm2d(256)
    conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))
    
    blocks = [
        conv1, relu, conv2, relu, conv3, relu, conv4, relu, 
        conv5, relu, conv6, pool, relu, conv7, bn1, relu, conv8, relu
    ]
    return nn.Sequential(*blocks)

class CRNN(nn.Module):
    def __init__(self, number_class_symbols):
        super().__init__()
        self.feature_extractor = CNN_backbone()
        self.avg_pool1 = nn.AdaptiveAvgPool2d((100, 100))
        self.bilstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, 
                             batch_first=True, bidirectional=True)
        self.avg_pool2 = nn.AdaptiveAvgPool2d((64, 64))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64, 32)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, number_class_symbols)

    def forward(self, x):
        x = self.feature_extractor(x)
        
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool1(x)
        x = x.transpose(1, 2)

        x, _ = self.bilstm(x)
        
        x = self.avg_pool2(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))
        x = self.gelu(x)    
        x = self.dropout(x) 
        x = self.fc2(x)
        return x 

def build_model(num_classes):
    """Model Setup"""
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    backbone = model.features
    fvit = timm.create_model('fastvit_sa12', pretrained=True)
    transformer_blocks = fvit.stages[3].blocks

    model = nn.Sequential(
        backbone,                               # Output: [Batch, 960, 7, 7]
        nn.Conv2d(960, 512, kernel_size=1),     # Channel adjustment to 512
        transformer_blocks,                     # Pure attention blocks (The Transformer part)
        nn.AdaptiveAvgPool2d(1),                # Spatial reduction to [Batch, 512, 1, 1]
        nn.Flatten(),                           # Flatten to [Batch, 512]
        nn.Linear(512, num_classes)             # Final classification head
    )
    model = CRNN(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(device)
