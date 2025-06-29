# SwinIR: Image Restoration Using Swin Transformer

This repository is the official PyTorch implementation of SwinIR: Image Restoration Using Shifted Window Transformer
([arxiv](https://arxiv.org/pdf/2108.10257.pdf), [supp](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0)). SwinIR ahcieves **state-of-the-art performance** in
- bicubic/lighweight/real-world image SR
- grayscale/color image denoising
- JPEG compression artifact reduction

</br>

:rocket:  :rocket:  :rocket: **News**:
 - *Aug. 26, 2021: See our recent work on real-world image SR: [a pratical degrdation model BSRGAN, ICCV2021](https://github.com/cszn/BSRGAN)*
 - *Aug. 26, 2021: See our recent work on generative modelling of image SR and image rescaling: [normalizing-flow-based HCFlow, ICCV2021](https://github.com/JingyunLiang/HCFlow)*
 - *Aug. 26, 2021: See our recent work on blind SR kernel estimation: [spatially variant kernel estimation (MANet, ICCV2021)](https://github.com/JingyunLiang/MANet) and [unsupervised kernel estimation (FKP, CVPR2021)](https://github.com/JingyunLiang/FKP)*

---

> Image restoration is a long-standing low-level vision problem that aims to restore high-quality images from low-quality images (e.g., downscaled, noisy and compressed images). While state-of-the-art image restoration methods are based on convolutional neural networks, few attempts have been made with Transformers which show impressive performance on high-level vision tasks. In this paper, we propose a strong baseline model SwinIR for image restoration based on the Swin Transformer. SwinIR consists of three parts: shallow feature extraction, deep feature extraction and high-quality image reconstruction. In particular, the deep feature extraction module is composed of several residual Swin Transformer blocks (RSTB), each of which has several Swin Transformer layers together with a residual connection. We conduct experiments on three representative tasks: image super-resolution (including classical, lightweight and real-world image super-resolution), image denoising (including grayscale and color image denoising) and JPEG compression artifact reduction. Experimental results demonstrate that SwinIR outperforms state-of-the-art methods on different tasks by up to 0.14~0.45dB, while the total number of parameters can be reduced by up to 67%.
><p align="center">
  <img width="800" src="figs/SwinIR_archi.png">
</p>



#### Contents

1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)
1. [Citation](#Citation)
1. [License and Acknowledgement](#License-and-Acknowledgement)


### Training


Used training and testing sets can be downloaded as follows:

| Task                 | Training Set | Testing Set|       
| :---                 | :---:        |     :---:      |
| classical/lightweight image SR          | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) or DIV2K +[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) | Set5 + Set14 + BSD100 + Urban100 + Manga109 [download all](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) |
| real-world image SR          | SwinIR-M (middle size): [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) +[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [OST](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip) (10324 images, sky,water,grass,mountain,building,plant,animal) <br /> SwinIR-L (large size): DIV2K + Flickr2K + OST + [WED](ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) + [FFHQ](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL) (first 2000 images, face) + Manga109 (manga) + [SCUT-CTW1500](https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip) (first 100 training images, texts) <br /><br />  ***We use the degradation model proposed in [BSRGAN, ICCV2021](https://github.com/cszn/BSRGAN)** | [RealSRSet](https://github.com/cszn/BSRGAN/tree/main/testsets/RealSRSet) | 
| color/grayscale image denoising      | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) |  grayscale: Set12 + BSD68 + Urban100 <br />  color: CBSD68 + Kodak24 + McMaster + Urban100 [download all](https://github.com/cszn/FFDNet/tree/master/testsets) | 
| JPEG compression artifact reduction  | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) |  grayscale: Classic5 +LIVE1 [download all](https://github.com/cszn/DnCNN/tree/master/testsets) |


<!--
| Task                 | Training Set | Testing Set|        Pretrained Model and Visual Results of SwinIR     | 
| :---                 | :---:        |     :---:      |:---:      |
| image denoising (real)      | [SIDD-Medium-sRGB](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) (320 images, [preprocess]()) + [RENOIR](http://ani.stat.fsu.edu/~abarbu/Renoir.html) (221 images, [preprocess](https://github.com/zsyOAOA/DANet/blob/master/datasets/preparedata/Renoir_big2small_all.py)) + [Poly](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset) (40 images in ./OriginalImages) |    [SIDD validation set](https://drive.google.com/drive/folders/1S44fHXaVxAYW3KLNxK41NYCnyX9S79su) (1280 patches, identical to official [.mat](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php) version) +  [DND](https://noise.visinf.tu-darmstadt.de/downloads/) (pre-defined 100 patches of 50 images, [online eval](https://noise.visinf.tu-darmstadt.de/submit/)) + [Nam](https://www.dropbox.com/s/24kds7c436i5i11/real_image_noise_dataset.zip?dl=0) (random 100 patches of 17 images, [preprocess](https://github.com/zsyOAOA/DANet/blob/master/datasets/preparedata/Nam_patch_prepare.py))|[download model]() [download results]() |
| image deblurring (synthetic)   | [GoPro](https://drive.google.com/drive/folders/1AsgIP9_X0bg0olu2-1N6karm2x15cJWE) (2103 training images)  |  [GoPro](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf) (1111 images) + [HIDE](https://drive.google.com/drive/folders/1nRsTXj4iTUkTvBhTcGg8cySK8nd3vlhK) (2050 images) + [RealBlur_J](https://drive.google.com/drive/folders/1KYtzeKCiDRX9DSvC-upHrCqvC4sPAiJ1) (real blur, 980 images) + [RealBlur_R](https://drive.google.com/drive/folders/1EwDoajf5nStPIAcU4s9rdc8SPzfm3tW1) (real blur, 980 images) | [download model]() [download results]()|
| image deraining (synthetic)  | [Multiple datasets](https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe) (13711 training images, see Table 1 of [MPRNet](https://github.com/swz30/MPRNet) for details.)  |  Rain100H (100 images) + Rain100L (100 images) + Test100 (100 images) + Test2800 (2800 images) + Test1200 (1200 images), [download all](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs)  | [download model]() [download results]()|

Note: above datasets may come from the official release or some awesome collections ([BasicSR](https://github.com/xinntao/BasicSR), [MPRNet](https://github.com/swz30/MPRNet)).

-->

The training code will be put in [KAIR](https://github.com/cszn/KAIR).

## Testing (without preparing datasets)
For your convience, we provide some example datasets (~20Mb) in `/testsets`. 
If you just want codes, downloading `models/network_swinir.py`, `utils/util_calculate_psnr_ssim.py` and `main_test_swinir.py` is enough.
Download [pretrained models](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) and put them in `model_zoo/swinir`, then run following commands:


```bash
# 001 Classical Image Super-Resolution (middle size)
# (when model is trained on DIV2K, use patch_size=48)
python main_test_swinir.py --task classical_sr --scale 2 --patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 3 --patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth --folder_lq testsets/Set5/LR_bicubic/X3 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 4 --patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 8 --patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth --folder_lq testsets/Set5/LR_bicubic/X8 --folder_gt testsets/Set5/HR

# (when model is trained on DIV2K+Flickr2K, use patch_size=64)
python main_test_swinir.py --task classical_sr --scale 2 --patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 3 --patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth --folder_lq testsets/Set5/LR_bicubic/X3 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 4 --patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task classical_sr --scale 8 --patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth --folder_lq testsets/Set5/LR_bicubic/X8 --folder_gt testsets/Set5/HR


# 002 Lightweight Image Super-Resolution (small size)
python main_test_swinir.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task lightweight_sr --scale 3 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth --folder_lq testsets/Set5/LR_bicubic/X3 --folder_gt testsets/Set5/HR
python main_test_swinir.py --task lightweight_sr --scale 4 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR


# 003 Real-World Image Super-Resolution
# (middle size)
python main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq testsets/RealSRSet+5images

# (larger size + trained on more datasets)
# python main_test_swinir.py --task real_sr --scale 4 --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq testsets/RealSRSet+5images


# 004 Grayscale Image Deoising (middle size)
python main_test_swinir.py --task gray_dn --noise 15 --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth --folder_gt testsets/Set12
python main_test_swinir.py --task gray_dn --noise 25 --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth --folder_gt testsets/Set12
python main_test_swinir.py --task gray_dn --noise 50 --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth --folder_gt testsets/Set12


# 005 Color Image Deoising (middle size)
python main_test_swinir.py --task color_dn --noise 15 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth --folder_gt testsets/McMaster
python main_test_swinir.py --task color_dn --noise 25 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth --folder_gt testsets/McMaster
python main_test_swinir.py --task color_dn --noise 50 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth --folder_gt testsets/McMaster


# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
python main_test_swinir.py --task jpeg_car --jpeg 10 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth --folder_gt testsets/classic5
python main_test_swinir.py --task jpeg_car --jpeg 20 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth --folder_gt testsets/classic5
python main_test_swinir.py --task jpeg_car --jpeg 30 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth --folder_gt testsets/classic5
python main_test_swinir.py --task jpeg_car --jpeg 40 --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth --folder_gt testsets/classic5

```

***All visual results of SwinIR can be downloaded [here](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0)**. 

*Large size real-world image SR model (`003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`) will be released later.

---

## Results
We achieved state-of-the-art performance on classical/lightweight/real-world image SR, grayscale/color image denoising and JPEG compression artifact reduction. Detailed results can be found in the [paper](https://arxiv.org/abs/2108.10257).

<details>
<summary>Classical Image Super-Resolution (click me)</summary>
<p align="center">
  <img width="900" src="figs/classic_image_sr.png">
  <img width="900" src="figs/classic_image_sr_visual.png">
</p>
</details>

<details>
<summary>Lightweight Image Super-Resolution</summary>
<p align="center">
  <img width="900" src="figs/lightweight_image_sr.png">
</p>
</details>

<details>
<summary>Real-World Image Super-Resolution</summary>
<p align="center">
  <img width="900" src="figs/real_world_image_sr.png">
</p>
</details>

<details>
<summary>Grayscale Image Deoising</summary>
<p align="center">
  <img width="900" src="figs/gray_image_denoising.png">
</p>
</details>

<details>
<summary>Color Image Deoising</summary>
<p align="center">
  <img width="900" src="figs/color_image_denoising.png">
</p>
</details>

<details>
<summary>JPEG Compression Artifact Reduction</summary>
<p align="center">
  <img width="900" src="figs/jepg_compress_artfact_reduction.png">
</p>
</details>



## Citation
    @article{liang2021swinir,
        title={SwinIR: Image Restoration Using Swin Transformer},
        author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
        journal={arXiv preprint arXiv:2108.10257}, 
        year={2021}
    }

## License and Acknowledgement
This project is released under the Apache 2.0 license. The codes are heavily based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer). We also refer to codes in [KAIR](https://github.com/cszn/KAIR) and [BasicSR](https://github.com/xinntao/BasicSR). Please also follow their licenses. Thanks for their awesome works.
