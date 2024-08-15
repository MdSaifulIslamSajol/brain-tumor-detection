#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 01:54:39 2024

@author: saiful
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data provided
data = {
    'Model': ['Swinv2', 'ViT', 'ConvNextV2', 'Cvt', 'EfficientFormer', 'PvtV2', 'MobileViTV2', 'ResNet50', 'VGG16', 'MobileNet', 'GoogleNet', 'EfficientNetB0'],
    'Test Accuracy': [95.4323, 96.4111, 90.7015, 84.5024, 90.8646, 93.3116, 88.9070, 96.9005, 90.0489, 93.8010, 93.1485, 92.0065],
    'Precision': [0.9547, 0.9570, 0.9034, 0.8752, 0.9074, 0.9252, 0.8776, 0.9635, 0.8948, 0.9302, 0.9292, 0.9166],
    'Recall': [0.9457, 0.9678, 0.9226, 0.8040, 0.8900, 0.9365, 0.8799, 0.9706, 0.9060, 0.9353, 0.9194, 0.9078],
    'F1 Score': [0.9494, 0.9616, 0.9049, 0.8221, 0.8971, 0.9293, 0.8785, 0.9664, 0.8953, 0.9326, 0.9220, 0.9114]
}

# Create DataFrame
df = pd.DataFrame(data)

# Sort by Test Accuracy in decreasing order
df_sorted = df.sort_values(by='Test Accuracy', ascending=False)

# Convert Precision, Recall, and F1 Score into percentile values (0-100)
df_sorted['Precision (%)'] = df_sorted['Precision'] * 100
df_sorted['Recall (%)'] = df_sorted['Recall'] * 100
df_sorted['F1 Score (%)'] = df_sorted['F1 Score'] * 100

# Update model name to match the color map key
df_sorted['Model'] = df_sorted['Model'].replace({'EfficientNet_b0': 'EfficientNetB0'})

# Define new metrics list for percentile values
percentile_metrics = ['Test Accuracy', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']

# Color assignments for each model based on the provided image
color_map = {
    'VGG16': '#cab2d6',
    'MobileNet': '#6a3d9a',
    'EfficientNetB0': '#d68a52',
    'EfficientFormer': '#fb9a99',
    'GoogleNet': '#e4a882',
    'ConvNextV2': '#b2df8a',
    'ResNet50': '#ff7f00',
    'PvtV2': '#e31a1c',
    'MobileViTV2': '#fdbf6f',
    'ViT': '#1f78b4',
    'Cvt': '#33a02c',
    'Swinv2': '#a6cee3'
}

# Plot each metric in a 2x2 subplot with specific colors for each model
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

ylim = (70, 100)

for i, metric in enumerate(percentile_metrics):
    ax = axes[i//2, i%2]
    bars = ax.bar(df_sorted['Model'], df_sorted[metric], color=[color_map[model] for model in df_sorted['Model']])
    ax.set_title(metric, fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticklabels(df_sorted['Model'], rotation=45, fontsize=14)
    ax.grid(True, axis='y')

plt.tight_layout()

# Save plots with updated colors and y-limits
png_file_path = '/mnt/data/model_performance_metrics.png'
svg_file_path = '/mnt/data/model_performance_metrics.svg'
plt.savefig(png_file_path, format='png', dpi=300)
plt.savefig(svg_file_path, format='svg', dpi=300)

# Show plot
plt.show()
