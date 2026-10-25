import time
import torch
import torch.nn as nn
from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D

torch.manual_seed(0)


def r2plus1d_34(num_classes=487):
    model = VideoResNet(block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[3, 4, 6, 3],
                         stem=R2Plus1dStem)
    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    # Caffe2-vs-PyTorch architecture fix from moabitcoin/ig65m-pytorch
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9
    return model


model = r2plus1d_34(num_classes=487)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"R(2+1)D-34 params: {n_params:,} ({n_params/1e6:.1f} M)")

# 8-frame snippet, 112x112, as GESTURES Table 1 reports for this checkpoint, should be 63.5M params, 512-d features
x = torch.randn(1, 3, 8, 112, 112)  # (B, C, T, H, W)

N_WARMUP = 3
N_REPS = 20

with torch.no_grad():
    for _ in range(N_WARMUP):
        model(x)
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - t0)

times.sort()
print(f"\n8-frame 112x112 snippet forward pass (batch=1, CPU, {torch.get_num_threads()} threads):")
print(f"  mean:   {sum(times)/len(times)*1e3:.1f} ms")
print(f"  median: {times[len(times)//2]*1e3:.1f} ms")
print(f"  min:    {times[0]*1e3:.1f} ms")
print(f"  max:    {times[-1]*1e3:.1f} ms")

from fvcore.nn import FlopCountAnalysis
flops = FlopCountAnalysis(model, x)
flops.unsupported_ops_warnings(False)
flops.uncalled_modules_warnings(False)
print(f"\nFLOPs (single 8-frame 112x112 forward pass): {flops.total()/1e9:.3f} GFLOPs")
