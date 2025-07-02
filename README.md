## Autoencoder-based Onboard Image Segmentation

This repository contains the implementation of a lightweight semantic segmentation system based on autoencoders, designed to operate **onboard a moving vehicle** in **rural environments**.

### Project Overview

The goal is to classify each pixel of an image (acquired by an onboard RGB camera) into one of the following 8 semantic classes:

* `0` – class 0
* `1` – class 1
* `2` – class 2
* `3` – class 3
* `4` – class 4
* `5` – class 5
* `6` – class 6
* `7` – class 7
* `8` – class 8

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

