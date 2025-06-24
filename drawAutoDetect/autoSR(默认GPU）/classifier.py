import cv2
from PIL import Image
import numpy as np

def classify_image(img: Image.Image) -> str:
    # 简单示例：根据图片特征判断类型，实际可用CNN/ViT等模型
    arr = np.array(img)
    h, w = arr.shape[:2]
    if h < 128 or w < 128:
        return 'noise'
    # 人脸检测
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        return 'face'
    # 这里可接入更复杂的分类模型
    # 示例：如果图片色彩鲜艳且有大面积纯色块，判为anime
    if arr.std() > 60 and np.mean(arr) > 100:
        return 'anime'
    # 示例：如果图片为人脸，可用人脸检测库
    # TODO: 接入人脸检测
    # 默认返回通用
    return 'general' 