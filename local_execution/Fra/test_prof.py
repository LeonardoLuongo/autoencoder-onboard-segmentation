# %%
# from google.colab import drive
# drive.mount('/content/drive')

# import os
# os.chdir("/content/drive/My Drive/Didattica/ML/exam_2024-2025/project_work_segmentazione_off-road/test")

# %% [markdown]
# #IMPLEMENT HERE YOUR CODE



# %% [markdown]
# ##IMPLEMENT HERE THE FUNCTION TO LOAD YOUR MODEL
# For example, here we use a single convolutional layer.

from torch.nn import Sequential, Conv2d
def load_model():
  return Sequential(Conv2d(in_channels=3, out_channels=9, kernel_size=1))

# %% [markdown]
# ##IMPLEMENT HERE YOUR PREDICT FUNCTION

import torch
def predict(model, X):
  model.eval()
  with torch.no_grad():

    # pre-processing
    X = torch.from_numpy(X).transpose(1,2).transpose(0,1).unsqueeze(0).float()
    X /= 255.0

    # forward function
    Y = model(X)

    # post-processing
    Y = Y.squeeze(0).transpose(0,1).transpose(1,2)
    Y = Y.argmax(dim=-1).squeeze(0).numpy().astype(np.uint8)
  return Y

# %% [markdown]
# #DO NOT MODIFY THE CODE BELOW!
# 
# This is exactly the code we run for the final test.
# 
# After implementing the previous functions, run this code to verify that it works

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Metrics
def compute_iou(mask1, mask2, label):
  intersection = np.sum((mask1 == label) & (mask2 == label))
  union = np.sum((mask1 == label) | (mask2 == label))
  if union == 0:
    return np.nan
  return intersection / union
def compute_all_iou(mask1, mask2, num_labels=8):
  iou_scores = np.zeros((num_labels))
  for label in range(num_labels):
    iou = compute_iou(mask1, mask2, label+1) # we skip the background label
    iou_scores[label] = iou
  return iou_scores


# Run YOUR LOAD_MODEL FUNCTION
model = load_model()

# Main loop
test_dir = "./train"  # we will change this path with that of the private test set directory
samples = os.listdir(test_dir)
IOUs = np.zeros((len(samples), 8))
verbose = False

for i, subdir in tqdm(enumerate(samples), desc="Processing samples"):
    subdir_path = os.path.join(test_dir, subdir)

    if os.path.isdir(subdir_path):
        # Get the data paths
        rgb_path = os.path.join(subdir_path, 'rgb.jpg')
        labels_path = os.path.join(subdir_path, 'labels.png')

        if os.path.exists(rgb_path) and os.path.exists(labels_path):
            if verbose:
                print(f"Processing subdirectory: {subdir}")

            try:  # ATTENTION: any error occurring in this try-catch means that the corresponding IOUs are evaluated as ZERO

                # Open images
                rgb_image = Image.open(rgb_path)
                rgb_array = np.asarray(rgb_image).copy()
                labels_image = Image.open(labels_path).copy()
                labels_array = np.asarray(labels_image)
                if verbose:
                    print(f"  Loaded {rgb_path} and {labels_path}")

                # Run YOUR PREDICT FUNCTION
                predicted_labels_array = predict(model, rgb_array)

                # Evaluate the IOU metric
                IOUs[i,:] = compute_all_iou(labels_array, predicted_labels_array)

                if verbose:
                    labels_vals = np.unique(np.asarray(labels_image))
                    print(f"  Unique labels values: {labels_vals}")
                    predicted_labels_vals = np.unique(np.asarray(predicted_labels_array))
                    print(f"  Unique predicted labels values: {predicted_labels_vals}")

                    plt.subplot(1, 3, 1)
                    plt.imshow(rgb_image)
                    plt.subplot(1, 3, 2)
                    plt.imshow(labels_image)
                    plt.subplot(1, 3, 3)
                    plt.imshow(predicted_labels_array)
                    plt.show()

                rgb_image.close()
                labels_image.close()

            except FileNotFoundError:
                print(f"  Error: Could not find image files in {subdir_path}")
            except Exception as e:
                print(f"  Error processing images in {subdir_path}: {e}")
        else:
            print(f"  Skipping subdirectory {subdir}: rgb.jpg or labels.png not found.")

score = np.nanmean(IOUs)
print(f"\nFinal competition score: {score}")

# %% [markdown]
# More information useful for

import numpy as np
np.set_printoptions(precision=3, suppress=True)
print(f"All IOUs:\n{IOUs}")
print("Average IOUs for each:")
print(f"- class: {np.nanmean(IOUs, 0)}")
print(f"- image: {np.nanmean(IOUs, 1)}")


