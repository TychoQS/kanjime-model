import sys, os, types, torch, importlib.util
import torch.nn.functional as F
from thop import profile

# Changing current path and adding to searchable modules routes
REPO = os.path.join(os.path.dirname(__file__), '..', 'models', 'fast')
os.chdir(REPO)
sys.path.insert(0, '.')

from models.builder import build_model
from models.utils import rep_model_convert

# Importing model config file
spec = importlib.util.spec_from_file_location("cfg", "config/fast/msra/fast_base_msra_736_finetune_ic17mlt.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Transforming dict to dict object required in build_model function
DictObj = type('DictObj', (dict,), {'__getattr__': dict.__getitem__})
to_cfg  = lambda d: DictObj({k: to_cfg(v) for k, v in d.items()}) if isinstance(d, dict) else d

# Building model and converting model
model = build_model(to_cfg(mod.model))
model = rep_model_convert(model)

# Removing head outputs
model.det_head.get_results = lambda *a, **kw: {}
model.det_head.loss        = lambda *a, **kw: {}

# Defining run configuration
run_cfg = type('cfg', (), {'debug': False, 'report_speed': False,
                           'test_cfg': type('test_cfg', (), {'scale': 1})()})()
# Getting stats
model.eval()

dummy    = torch.randn(1, 3, 1920, 1080)
flops, _ = profile(model, inputs=(dummy, None, None, None, None, None, run_cfg), verbose=False)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.save(model.state_dict(), '/tmp/fast_b_736.pth')

def forward_onnx(self, x):
    f = self.backbone(x)
    f = self.neck(f)
    return self.det_head(f)

model.forward = types.MethodType(forward_onnx, model)
torch.onnx.export(model, dummy, '/tmp/fast_b_736.onnx', opset_version=18)

print(f"Params    : {total/1e6:.2f}M")
print(f"FLOPs     : {flops/1e9:.2f}G")
print(f"pth       : {os.path.getsize('/tmp/fast_b_736.pth')/1e6:.1f} MB")
print(f"ONNX      : {os.path.getsize('/tmp/fast_b_736.onnx')/1e6:.1f} MB")
print(f"CSV:{total/1e6:.2f},{flops/1e9:.2f},{os.path.getsize('/tmp/fast_b_736.pth')/1e6:.1f},{os.path.getsize('/tmp/fast_b_736.onnx')/1e6:.1f}")