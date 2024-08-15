#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:18:41 2024

@author: saiful
"""

import matplotlib.pyplot as plt

# Short names for models and their inference times
models = [
    "Swinv2", "ViT", "ConvNextV2", "Cvt", "EfficientFormer", 
    "PvtV2", "MobileViTV2", "ResNet50", "VGG16", "MobileNet", 
    "GoogleNet", "EfficientNetB0"
]
inference_times = [
    0.002807, 0.001206, 0.000676, 0.001835, 0.000525, 
    0.000834, 0.000977, 0.000697, 0.000111, 0.000348, 
    0.000586, 0.000478
]

# Colors assigned to models in the previous plot
colors = [
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", 
    "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", 
    "#ffff99", "#b15928"
]

# Reversing the order of inference times
reversed_indices = sorted(range(len(inference_times)), key=lambda k: inference_times[k], reverse=True)
reversed_models = [models[i] for i in reversed_indices]
reversed_times = [inference_times[i] for i in reversed_indices]
reversed_colors = [colors[i] for i in reversed_indices]

# Plotting the data with specified requirements and font size 12
fig, ax = plt.subplots(figsize=(20, 10))
bars = ax.barh(reversed_models, reversed_times, color=reversed_colors)

# Adding labels and title
ax.set_xlabel("Inference Time (seconds)", fontsize=12)
ax.set_ylabel("Models", fontsize=12)
ax.set_title("Inference Time per Image for Different Models", fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)

# Removing the grid
ax.grid(False)

# Saving the plot in SVG and PNG formats with 300 dpi
svg_path = "/mnt/data/inference_times_plot_reversed_fontsize12.svg"
png_path = "/mnt/data/inference_times_plot_reversed_fontsize12.png"

plt.savefig(svg_path, format='svg', dpi=300)
plt.savefig(png_path, format='png', dpi=300)

# Displaying the plot
plt.show()

# Displaying paths
(svg_path, png_path)
