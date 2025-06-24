from PIL import Image
import numpy as np
import torch
import os
from gfpgan import GFPGANer

def process(img: Image.Image) -> Image.Image:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 权重文件下载：https://github.com/TencentARC/GFPGAN/releases
    weight_path = os.path.join(os.path.dirname(__file__), 'GFPGANv1.4.pth')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"未找到权重文件: {weight_path}\n请从 https://github.com/TencentARC/GFPGAN/releases 下载 GFPGANv1.4.pth 并放到 models 目录下")
    model = GFPGANer(
        model_path=weight_path,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=device
    )
    img_np = np.array(img.convert('RGB'))
    _, _, output = model.enhance(img_np, has_aligned=False, only_center_face=False)
    return Image.fromarray(output) 