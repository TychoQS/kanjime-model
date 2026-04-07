from torchvision import transforms

def get_base_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])