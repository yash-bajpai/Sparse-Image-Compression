# 🖼️ Sparse Image Compressor using DCT

This project provides a simple yet effective way to **compress images using Discrete Cosine Transform (DCT)** and **sparse matrix operations**. It supports both grayscale and color images.

---

## 📦 Features

- Compresses images by retaining only a fraction of significant DCT coefficients.
- Works on images in `.jpg`, `.png`, and other common formats.
- Supports grayscale and RGB color images.
- Block-wise DCT compression with adjustable block size.
- Reports compression statistics (original vs. compressed size).

---

## 🛠️ How It Works

1. Image is divided into `8x8` (or custom size) blocks.
2. 2D DCT is applied to each block.
3. Only top-k DCT coefficients are retained (based on `compression_ratio`).
4. The rest are zeroed out to create sparsity.
5. Inverse DCT is used to reconstruct the image.

---

## 🔧 Requirements

- Python 3.x
- `numpy`
- `scipy`
- `Pillow`

You can install dependencies using:

```bash
pip install -r requirements.txt
