#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 00:48:17 2024

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

# Convert seconds to microseconds
reversed_times_microseconds = [time * 1e6 for time in reversed_times]

# Plotting the data with specified requirements, font size 18, and printing inference values in microseconds
fig, ax = plt.subplots(figsize=(7, 3.92))  # Adjusted to 16:9 ratio for IEEE paper
bars = ax.barh(reversed_models, reversed_times_microseconds, color=reversed_colors)

# Adding labels and title
ax.set_xlabel("Inference Time (microseconds)", fontsize=18)
ax.set_ylabel("Models", fontsize=18)
ax.set_title("Inference Time per Image for Different Models", fontsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)

# Adding the inference values on top of each bar
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.0f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),  # 5 points offset to the right
                textcoords="offset points",
                ha='left', va='center', fontsize=20)

# Removing the grid
ax.grid(False)

# Adding legend with assigned colors
legend_labels = [f'{model}' for model in reversed_models]
handles = [plt.Rectangle((0,0),1,1, color=color) for color in reversed_colors]
ax.legend(handles, legend_labels, title="Models", title_fontsize=20, fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# Saving the plot in SVG and PNG formats with 300 dpi
svg_path = "/mnt/data/inference_times_plot_reversed_fontsize18_ieee_16_9.svg"
png_path = "/mnt/data/inference_times_plot_reversed_fontsize18_ieee_16_9.png"

plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

# Displaying the plot
plt.show()

# Displaying paths
(svg_path, png_path)


# Plotting the data with specified requirements, font size 12, and printing inference values in microseconds
fig, ax = plt.subplots(figsize=(7, 3.92))  # Adjusted to 16:9 ratio for IEEE paper
bars = ax.barh(reversed_models, reversed_times_microseconds, color=reversed_colors)

# Adding labels and title
ax.set_xlabel("Inference Time (microseconds)", fontsize=12)
ax.set_ylabel("Models", fontsize=12)
ax.set_title("Inference Time per Image for Different Models", fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)

# Adding the inference values on top of each bar
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.0f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),  # 5 points offset to the right
                textcoords="offset points",
                ha='left', va='center', fontsize=12)

# Removing the grid
ax.grid(False)

# Adding legend with assigned colors
legend_labels = [f'{model}' for model in reversed_models]
handles = [plt.Rectangle((0,0),1,1, color=color) for color in reversed_colors]
ax.legend(handles, legend_labels, title="Models", title_fontsize=20, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# Saving the plot in SVG and PNG formats with 300 dpi
svg_path = "/mnt/data/inference_times_plot_reversed_fontsize12_ieee_16_9.svg"
png_path = "/mnt/data/inference_times_plot_reversed_fontsize12_ieee_16_9.png"

plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

# Displaying the plot
plt.show()

# Displaying paths
(svg_path, png_path)
