import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out

def process(img: Image.Image) -> Image.Image:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = os.path.join(os.path.dirname(__file__), 'net.pth')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"未找到权重文件: {weight_path}\n请将 net.pth 放到 models 目录下")
    model = DnCNN(channels=3, num_of_layers=17)
    state_dict = torch.load(weight_path, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model.to(device)
    img_np = np.array(img).astype(np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=2)
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    img_tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        out_tensor = model(img_tensor)
    out_np = (out_tensor.squeeze().permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(out_np) 