## Autoencoder-based Onboard Image Segmentation

This repository contains the implementation of a lightweight semantic segmentation system based on autoencoders, designed to operate **onboard a moving vehicle** in **rural environments**.

### Project Overview

The goal is to classify each pixel of an image (acquired by an onboard RGB camera) into one of the following 8 semantic classes:

* `0` – Sky
* `1` – Rough trail
* `2` – Smooth trail
* `3` – Traversable grass
* `4` – High vegetation
* `5` – Non-traversable low vegetation
* `6` – Puddle
* `7` – Obstacle
* `8` – Background

### Model Architecture

The core of the system is a convolutional **autoencoder**, trained to map RGB input images to pixel-wise class probabilities. The network is optimized for onboard use, balancing **accuracy**, **inference speed**, and **memory footprint** (≤4GB GPU RAM during test).

### Features

* Customizable encoder-decoder architecture
* Train/validation split strategy
* Data augmentation pipeline
* Pixel-wise classification with `MxNx9` softmax output
* Evaluation using per-class **mean Intersection-over-Union (mIoU)**
* Google Colab compatibility (train/test notebooks)

### 📁 Repository Structure

```

```

### 🚀 How to Run



### 📊 Evaluation Metrics

