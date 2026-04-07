import segmentation_models_pytorch as smp
from .config import ENCODER_NAME, ENCODER_WEIGHTS

def get_unet_model():
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None
    )
    return model