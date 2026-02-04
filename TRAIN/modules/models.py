import torch
import torch.nn as nn
import torchvision.models as models

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

        # Loading model
        base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        # Getting features extractors
        self.backbone = base_model.features
        self.avgpool = base_model.avgpool

        self.feature_dim = base_model.classifier[3].in_features

        self.intermediate = nn.Sequential(
            base_model.classifier[0],
            base_model.classifier[1],
            base_model.classifier[2]
        )

        # Defining last layers (multiples heads)
        self.components_head = nn.Linear(self.feature_dim, num_components_classes)
        self.kanji_head = nn.Linear(self.feature_dim + num_components_classes, num_kanji_classes)

    def forward(self, x):
        # Passing the image through the network
        x = self.backbone(x)
        x = self.avgpool(x)
        
        # Flattten to pass to Linear layers
        x = torch.flatten(x, 1)
        
        x = self.intermediate(x)

        # Getting components predictions
        component_logits = self.components_head(x)

        # Combining with the output of classifiers 
        combined_info = torch.cat([x, component_logits], dim=1)

        # Getting kanji predictions
        kanji_logits = self.kanji_head(combined_info)
        
        return kanji_logits, component_logits