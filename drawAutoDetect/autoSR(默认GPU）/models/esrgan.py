import torch
import numpy as np
from PIL import Image
import os
from models.ESRGAN_arch import RRDBNet

def process(img: Image.Image) -> Image.Image:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = os.path.join(os.path.dirname(__file__), 'RRDB_ESRGAN_x4.pth')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"未找到ESRGAN权重文件: {weight_path}")
    # RRDBNet参数需与权重一致
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    # 预处理
    img = img.convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    out_np = (output.squeeze().permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype(np.uint8)
    return Image.fromarray(out_np) 