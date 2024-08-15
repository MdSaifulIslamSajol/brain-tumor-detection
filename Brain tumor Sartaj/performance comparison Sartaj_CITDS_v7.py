import matplotlib.pyplot as plt
import pandas as pd

# Data provided
data = {
    'Model': [
        'Swinv2', 'ViT', 'ConvNextV2', 
        'Cvt', 'EfficientFormer', 'PvtV2', 
        'MobileViTV2', 'ResNet50', 'VGG16', 
        'MobileNet', 'GoogleNet', 'EfficientNetB0'
    ],
    'Test Accuracy': [
        70.8122, 68.7817, 68.5279, 63.9594, 58.8832, 65.2284, 
        43.9086, 69.0355, 53.0457, 68.7817, 64.2132, 64.4670
    ],
    'Precision': [
        0.7677, 0.7654, 0.7615, 0.7124, 0.6992, 0.7336, 
        0.5111, 0.7768, 0.5548, 0.7302, 0.6983, 0.7599
    ],
    'Recall': [
        0.6981, 0.6648, 0.6597, 0.6309, 0.5680, 0.6167, 
        0.4608, 0.6688, 0.5394, 0.6758, 0.6356, 0.6314
    ],
    'F1 Score': [
        0.6670, 0.6493, 0.6575, 0.6292, 0.5448, 0.5975, 
        0.4318, 0.6482, 0.5391, 0.6670, 0.6100, 0.6108
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Sort by Test Accuracy in decreasing order
df_sorted = df.sort_values(by='Test Accuracy', ascending=False)

# Convert Precision, Recall, and F1 Score into percentile values (0-100)
df_sorted['Precision (%)'] = df_sorted['Precision'] * 100
df_sorted['Recall (%)'] = df_sorted['Recall'] * 100
df_sorted['F1 Score (%)'] = df_sorted['F1 Score'] * 100

# Define new metrics list for percentile values
percentile_metrics = ['Test Accuracy', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']

# Color assignments for each model
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

ylim = (40, 80)

for i, metric in enumerate(percentile_metrics):
    ax = axes[i//2, i%2]
    bars = ax.bar(df_sorted['Model'], df_sorted[metric], color=[color_map[model] for model in df_sorted['Model']])
    ax.set_title(metric, fontsize=24)
    ax.set_xlabel('Model', fontsize=18)
    ax.set_ylabel(metric, fontsize=18)
    ax.set_ylim(ylim)
    ax.set_xticklabels(df_sorted['Model'], rotation=60, fontsize=18)
    ax.grid(False)  # Turn off grid lines
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=15, clip_on=False)

plt.tight_layout()

# Save plots with values on top of bars and without grid lines
png_file_path = '/mnt/data/model_performance_metrics_v4.png'
svg_file_path = '/mnt/data/model_performance_metrics_v4.svg'
plt.savefig(png_file_path, format='png', dpi=300)
plt.savefig(svg_file_path, format='svg', dpi=300)

# Show plot
plt.show()
