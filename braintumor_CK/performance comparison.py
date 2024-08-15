#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:41:52 2024

@author: saiful
"""

import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    'Swinv2', 'ViT', 'ConvNextV2', 'Cvt', 'EfficientFormer', 'PvtV2', 
    'MobileViT', 'resnet50', 'vgg16', 'mobilenet', 'googlenet', 'efficientnet_b0'
]

# Training and Validation accuracies of epoch 1 for each model
train_accuracies = [90.5802, 89.4573, 93.1958, 77.9030, 92.6299, 91.2040, 89.0696, 93.5389, 90.1167, 91.8947, 91.3466, 90.5623]
test_accuracies = [96.0058, 96.7320, 95.4975, 97.8214, 97.7487, 95.2070, 96.0058, 96.2963, 95.4975, 97.3856, 95.7153, 96.8773]

# Sort models based on test accuracy in decreasing order
sorted_indices = np.argsort(test_accuracies)[::-1]
models = np.array(models)[sorted_indices]
train_accuracies = np.array(train_accuracies)[sorted_indices]
test_accuracies = np.array(test_accuracies)[sorted_indices]

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Bar colors
colors = plt.cm.tab20(np.linspace(0, 1, len(models)))

# Training Accuracy
ax[0].bar(models, train_accuracies, color=colors)
ax[0].set_title('Training Accuracy at Epoch 1')
ax[0].set_ylabel('Accuracy (%)')
ax[0].set_ylim(75, 100)
ax[0].set_xticklabels(models, rotation=45, ha='right')

# Test Accuracy
ax[1].bar(models, test_accuracies, color=colors)
ax[1].set_title('Test Accuracy at Epoch 1')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_ylim(75, 100)
ax[1].set_xticklabels(models, rotation=45, ha='right')

plt.tight_layout()


# Save the figure with 600 DPI
png_path_high_dpi = "/home/saiful/brain tumor/braintumor_CK/results/model_accuracies_high_dpi.png"
svg_path_high_dpi = "/home/saiful/brain tumor/braintumor_CK/results/model_accuracies_high_dpi.svg"
plt.savefig(png_path_high_dpi, dpi=600)
plt.savefig(svg_path_high_dpi, dpi=600)
plt.show()
