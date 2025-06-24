import os
import subprocess
from PIL import Image

def process(img: Image.Image) -> Image.Image:
    temp_in = 'waifu2x_temp_in.png'
    temp_out = 'waifu2x_temp_out.png'
    img.save(temp_in)
    exe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../waifu2x-ncnn-vulkan/waifu2x-ncnn-vulkan.exe'))
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"未找到waifu2x-ncnn-vulkan.exe: {exe_path}\n请下载并解压到指定目录")
    cmd = [
        exe_path,
        '-i', temp_in,
        '-o', temp_out,
        '-n', '2',        # 去噪等级 0-3
        '-s', '2',        # 放大倍数 2
        '-m', 'models-upconv_7_anime_style_art_rgb'  # 新版模型文件夹
    ]
    subprocess.run(cmd, check=True)
    out_img = Image.open(temp_out).copy()
    os.remove(temp_in)
    os.remove(temp_out)
    return out_img 