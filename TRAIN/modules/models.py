import torch
import timm

def build_model(num_classes):
    model = timm.create_model(
        'ghostnet_100', 
        pretrained=True,
        num_classes=num_classes,
        in_chans=1  
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(device)