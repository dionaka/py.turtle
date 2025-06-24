import os
from classifier import classify_image
from models import esrgan, waifu2x, gfpgan, dncnn
from PIL import Image
import argparse
from timm.layers import LayerNorm, DropPath
import numpy as np

MODEL_MAP = {
    'anime': (waifu2x.process, 'waifu2x 动漫超分与去噪'),
    'face': (gfpgan.process, 'GFPGAN 人脸修复'),
    'noise': (dncnn.process, 'DnCNN 去噪'),
    'general': (esrgan.process, 'ESRGAN 通用超分')
}

def main(input_path, output_path, model_type=None):
    img = Image.open(input_path)
    if model_type and model_type in MODEL_MAP:
        func, desc = MODEL_MAP[model_type]
        print(f"用户指定模型：{model_type}，调用{desc}")
        out_img = func(img)
    else:
        img_type = classify_image(img)
        if img_type in MODEL_MAP:
            func, desc = MODEL_MAP[img_type]
            print(f"识别为：{img_type}，调用{desc}")
            out_img = func(img)
        else:
            print(f"识别为：{img_type}，调用ESRGAN（默认）")
            out_img = esrgan.process(img)
    # 检查输出图片和输入图片是否有变化
    input_arr = np.array(img)
    output_arr = np.array(out_img)
    if input_arr.shape == output_arr.shape:
        diff = np.abs(input_arr.astype(np.int32) - output_arr.astype(np.int32))
        max_diff = diff.max()
        mean_diff = diff.mean()
        if max_diff == 0:
            print("【警告】输出图片与输入图片完全一致，模型可能未做任何处理！")
        else:
            print(f"图片已变化，最大像素差：{max_diff}，平均像素差：{mean_diff:.2f}")
    else:
        print("图片尺寸发生变化，模型已做处理。")
    out_img.save(output_path)
    print(f"处理完成，输出：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入图片路径')
    parser.add_argument('--output', default='output.png', help='输出图片路径')
    parser.add_argument('--model', default=None, help='指定模型类型（anime, face, photo, noise, general）')
    args = parser.parse_args()
    main(args.input, args.output, args.model) 