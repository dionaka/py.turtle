from PIL import Image
import numpy as np

# 读取原图
img = Image.open('picture/un.png').convert('RGB')
img_np = np.array(img).astype(np.float32)

# 添加高斯噪声
mean = 0
std = 25  # 噪声强度，数值越大噪声越明显
noise = np.random.normal(mean, std, img_np.shape)
noisy_img = img_np + noise

# 裁剪到合法像素范围
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# 保存加噪声的图片
Image.fromarray(noisy_img).save('picture/un_noisy.png')
print("已生成高斯噪声图片：picture/un_noisy.png")