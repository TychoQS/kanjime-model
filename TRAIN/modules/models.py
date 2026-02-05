import torch
import torch.nn as nn
import torchvision.models as models
import timm

def build_model(num_classes):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(device)

def build_multi_head_model(num_classes, components_classes):
    model = MultiHeadKanjiClassificator(num_classes, components_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

class MultiHeadKanjiClassificator(nn.Module):
    """A multihead model for inference Kanji and it's components"""
    
    def __init__(self, num_kanji_classes, num_components_classes):
        super().__init__()
        
        self.backbone = timm.create_model('ghostnet_100', pretrained=True, num_classes=0)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            self.feature_dim = dummy_output.shape[1]
        
        self.intermediate = nn.Identity()
        self.components_head = nn.Linear(self.feature_dim, num_components_classes)
        self.kanji_head = nn.Linear(self.feature_dim + num_components_classes, num_kanji_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.intermediate(x)
        component_logits = self.components_head(x)
        combined_info = torch.cat([x, component_logits], dim=1)
        kanji_logits = self.kanji_head(combined_info)
        return kanji_logits, component_logits