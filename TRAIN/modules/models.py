import torch
import torch.nn as nn
import timm

def build_multi_head_model(num_kanji_classes, num_radical_classes, num_stroke_classes):  
    model = MultiHeadKanjiClassificator(num_kanji_classes, num_radical_classes, num_stroke_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

class MultiHeadKanjiClassificator(nn.Module):
    """A multihead model for inference Kanji using predictions of number of strokes and radicals"""
    
    def __init__(self, num_kanji_classes, num_radical_classes, num_stroke_classes):
        super().__init__()
        
        self.backbone = timm.create_model('ghostnet_100', pretrained=True, num_classes=0)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            self.feature_dim = dummy_output.shape[1]
        
        self.intermediate = nn.Identity()
        self.radical_head = nn.Linear(self.feature_dim, num_radical_classes)
        self.strokes_head = nn.Linear(self.feature_dim, num_stroke_classes)
        self.kanji_head = nn.Linear(self.feature_dim + num_radical_classes + num_stroke_classes, num_kanji_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.intermediate(x)
        radical_logits = self.radical_head(x)
        strokes_logits = self.strokes_head(x)
        kanji_logits = self.kanji_head(torch.cat([x, radical_logits, strokes_logits], dim=1))
        return kanji_logits, radical_logits, strokes_logits