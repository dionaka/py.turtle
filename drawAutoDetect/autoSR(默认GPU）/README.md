# autoSR

## 项目简介
本项目实现自动识别图片类型，并根据图片内容调用最合适的AI模型进行超分辨率、去噪、人脸修复等处理。

支持的AI模型：
- SRGAN/ESRGAN：通用超分辨率//ok
- Waifu2x：动漫图片超分辨率与去噪//ok
- GFPGAN：人脸修复//ok
- DnCNN：图像去噪//okok(已验证）
- SwinIR：多任务图像修复 //autoSR/models/SwinIR-0.0/SwinIR-0.0/main_test_swinir.py，之前draw AutoDetect脚本里实现过这里就不加了，当时是用作超分放大，可以过去看看

## 项目结构
```
autoSR/
├── main.py                # 主程序入口，自动识别图片类型并分发处理
├── classifier.py          # 图片类型识别模块
├── models/                # 各AI模型调用封装
│   ├── esrgan.py
│   ├── waifu2x.py
│   ├── gfpgan.py
│   ├── dncnn.py
│   └── swinir.py
├── requirements.txt       # 依赖库
└── README.md              # 项目说明
```

## 主要依赖
- torch
- torchvision
- numpy
- pillow
- opencv-python
- 以及各模型的官方/第三方实现

## 使用方法
1. 安装依赖：`pip install -r requirements.txt`
2. 运行主程序：`python main.py --input picture/un_noisy.png --output yyy.png`（自动检测并对图片处理）
--model noise（可选方法）

## 可选方法
MODEL_MAP = {
    'anime': (waifu2x.process, 'waifu2x 动漫超分与去噪'), //ok
    'face': (gfpgan.process, 'GFPGAN 人脸修复'), //ok
    'noise': (dncnn.process, 'DnCNN 去噪'),  //ok
    'general': (esrgan.process, 'ESRGAN 通用超分')  //ok
}

主程序会自动识别图片类型，并调用最优模型处理，输出结果图片。 