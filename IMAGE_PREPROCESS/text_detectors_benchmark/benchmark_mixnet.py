import sys, os, torch, importlib.util
from thop import profile

# Changing current path and adding to searchable modules routes
REPO = os.path.join(os.path.dirname(__file__), '..', 'models', 'mixnet')
os.chdir(REPO)
sys.path.insert(0, '.')

# Importing model config file
spec = importlib.util.spec_from_file_location("config", "cfglib/config.py")
cfg_mod = importlib.util.module_from_spec(spec)
sys.modules["config"] = cfg_mod
sys.modules["cfglib.config"] = cfg_mod
spec.loader.exec_module(cfg_mod)

# Configuring model
cfg = cfg_mod.config
cfg.resume       = False
cfg.onlybackbone = True
cfg.mid          = False
cfg.embed        = False
cfg.device       = 'cpu'
cfg.test_size    = [640, 1024]

from network.textnet import TextNet

# Building model
model = TextNet(backbone='FSNet_M', is_training=False)
model.eval()

# Getting parameters
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
dummy     = torch.randn(1, 3, 640, 1024)
flops, _  = profile(model, inputs=({'img': dummy},), verbose=False)

torch.save(model.state_dict(), '/tmp/mixnet.pth')

class MixNetExport(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m
    def forward(self, x):
        return self.model({'img': x})

torch.onnx.export(MixNetExport(model), dummy, '/tmp/mixnet.onnx', opset_version=18, dynamo=False)

print(f"Params    : {total/1e6:.2f}M")
print(f"FLOPs     : {flops/1e9:.2f}G")
print(f"pth       : {os.path.getsize('/tmp/mixnet.pth')/1e6:.1f} MB")
print(f"ONNX      : {os.path.getsize('/tmp/mixnet.onnx')/1e6:.1f} MB")