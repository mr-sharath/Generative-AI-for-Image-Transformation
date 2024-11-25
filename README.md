# 🎨 ESRGAN: Enhanced Super-Resolution GAN from Scratch

Welcome to the **ESRGAN** (Enhanced Super-Resolution Generative Adversarial Network) project! This repository provides an **implementation of ESRGAN from scratch** using PyTorch. It's a powerful model designed to upscale low-resolution images into high-resolution, realistic visuals. If you're interested in **image super-resolution** and GANs, you've come to the right place!

This project is built based on the **ESRGAN research paper**, which extends upon SRGAN by introducing **Residual-in-Residual Dense Blocks (RRDB)** and other advanced techniques like **relativistic adversarial loss**. Check out the original ESRGAN paper for a deeper understanding, but here we’ll focus on the **model architecture** and how to implement it.

---

## 🚀 Features

- **ESRGAN Model from Scratch**: Fully implemented ESRGAN generator and discriminator models using PyTorch.
- **Advanced Blocks**: Incorporates **RRDB** for better image super-resolution.
- **Pre-Trained Weights**: Load and use official pre-trained weights without retraining.
- **Efficient Upsampling**: Uses **nearest-neighbor upsampling** instead of pixel shuffle.
- **Highly Configurable**: Modify number of channels, layers, and upsampling factors easily.
  
---

## 📜 Paper Reference

If you're new to ESRGAN, we highly recommend reading the following papers to understand the theoretical concepts before diving into the code:

- **[SRGAN Paper](https://arxiv.org/abs/1609.04802)** - The foundation of ESRGAN, introducing super-resolution GANs.
- **[ESRGAN Paper](https://arxiv.org/abs/1809.00219)** - The improved GAN architecture that this project is based on.

---

## 🛠️ Implementation Details

The architecture includes several key components:

### 🧠 **Generator**:
The **generator** network utilizes **Residual-in-Residual Dense Blocks (RRDB)** to generate high-quality images. It consists of:
- **Conv blocks** with **Leaky ReLU** activation (no batch norm!).
- **RRDB** for better **feature learning**.
- **Nearest-neighbor upsampling** for efficient high-resolution generation.

### 🧠 **Discriminator**:
The **discriminator** follows a VGG-like architecture to classify real vs. fake images. It’s designed to work with **96x96** image inputs and uses **average pooling** to support variable input sizes.

### 🖼️ **Upsample Block**:
- Uses **nearest-neighbor upsampling** instead of the traditional **pixel shuffle**.
- Includes adjustable scale factors for **flexible resolution scaling**.


![ESRGAN](https://github.com/user-attachments/assets/23e6ec96-ac66-40e5-a363-6d6bc288a882)

---

## 📦 Pre-Trained Weights

Training ESRGAN from scratch can be time-consuming. Instead, you can use **pre-trained weights** for quick results! You can easily load these weights into your custom model without hassle.

```python
# Load pre-trained weights
model.load_state_dict(torch.load('pretrained_weights.pth'))
```

---

## 🖥️ Installation

To get started with ESRGAN, clone this repo and install the required dependencies:

```bash
git clone https://github.com/mr-sharath/esrgan-from-scratch.git
cd esrgan-from-scratch
pip install -r requirements.txt
```

---

## 📚 How to Use

### Preparing Input and Output Folders:
- Place your low-resolution **input images** in the `test_images` folder.
- The generated **high-resolution output images** will be saved in the `saved` folder after the model processes them.

### Training:
You can modify and train the model by running the `train.py` script. This script sets up the training pipeline, including loss functions and optimizers.

```bash
python train.py --epochs 100 --batch_size 16
```

### Testing:
Test your model on new images using `train.py`. You can input low-resolution images, and the model will upscale them.

```bash
python train.py --input test_images/low_res_image.png --output saved/high_res_image.png
```

### Model Summary:
Here's a quick summary of how the generator works:

1. **Input**: Takes low-res images (e.g., 24x24 pixels).
2. **RRDB Block**: Passes through **RRDB blocks** for feature enhancement.
3. **Upsample**: Upscales the image using nearest-neighbor interpolation.
4. **Output**: Produces a high-res image (e.g., 96x96 pixels).

---

## 📈 Results

ESRGAN achieves superior image quality over SRGAN by focusing on **perceptual quality** rather than pixel-wise accuracy. Below are some comparison results:

| Method   | PSNR | SSIM | Perceptual Quality |
|----------|------|------|--------------------|
| SRGAN    | 26.0 | 0.75 | 🟠                 |
| **ESRGAN** | 26.6 | 0.79 | 🟢                 |

---

## 🙌 Acknowledgements

Shout out to the creators of the ESRGAN paper and all the open-source contributors to **PyTorch** and GAN research! Also, thanks to **Francesco** for insights on optimizing PyTorch imports!

---

## 🔗 Connect with Me

Feel free to open issues or make pull requests if you want to contribute! If you have any questions, reach out to me via [GitHub](https://github.com/mr-sharath) or connect on LinkedIn.

---

⭐ **If you like this project, don't forget to give it a star!** ⭐
