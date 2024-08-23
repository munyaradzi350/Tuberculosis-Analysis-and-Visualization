import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_images(image_paths, label, image_size=(128, 128)):
    images = []
    labels = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.resize(img, image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Function to display sample images
def display_samples(images, labels, category):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(f'{category} {i+1}')
        plt.axis('off')
    plt.show()

# Load a few sample images from each category
num_samples = 5
# Update dataset_path to reflect the correct location of your data folder
dataset_path = '../data/'

# Construct paths to normal and tuberculosis folders
normal_path = os.path.join(dataset_path, 'normal')
tb_path = os.path.join(dataset_path, 'tuberculosis')

# Debug prints to check absolute paths
print("Absolute path to normal folder:", os.path.abspath(normal_path))
print("Absolute path to tuberculosis folder:", os.path.abspath(tb_path))

# Check if the directories exist
if not os.path.exists(normal_path) or not os.path.exists(tb_path):
    raise FileNotFoundError(f'Could not find directories: {normal_path} or {tb_path}')

# List PNG files in the directories
normal_images = [img for img in os.listdir(normal_path) if img.endswith('.png')][:num_samples]
tb_images = [img for img in os.listdir(tb_path) if img.endswith('.png')][:num_samples]

# Create full paths to each sample image
normal_sample_paths = [os.path.join(normal_path, img) for img in normal_images]
tb_sample_paths = [os.path.join(tb_path, img) for img in tb_images]

# Load and preprocess images
normal_samples, normal_labels = load_images(normal_sample_paths, label=0)
tb_samples, tb_labels = load_images(tb_sample_paths, label=1)

# Display samples
print("Normal Samples:")
display_samples(normal_samples, normal_labels, 'Normal')

print("Tuberculosis Samples:")
display_samples(tb_samples, tb_labels, 'Tuberculosis')
