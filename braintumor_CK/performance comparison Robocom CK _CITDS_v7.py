import matplotlib.pyplot as plt
import pandas as pd

# New data provided
data = {
    'Model': [
        'Swinv2', 'ViT', 'ConvNextV2', 
        'Cvt', 'EfficientFormer', 'PvtV2', 
        'MobileViTV2', 'ResNet50', 'VGG16', 
        'MobileNet', 'GoogleNet', 'EfficientNetB0'
    ],
    'Test Accuracy': [
        95.2796, 95.7879, 97.6761, 91.2128, 98.4023, 97.8940, 
        96.8046, 96.0784, 95.6427, 95.6427, 97.6761, 96.8046
    ],
    'Precision': [
        0.9565, 0.9552, 0.9755, 0.9153, 0.9829, 0.9766, 
        0.9656, 0.9581, 0.9591, 0.9525, 0.9740, 0.9658
    ],
    'Recall': [
        0.9461, 0.9565, 0.9759, 0.9185, 0.9835, 0.9791, 
        0.9664, 0.9609, 0.9499, 0.9568, 0.9779, 0.9671
    ],
    'F1 Score': [
        0.9494, 0.9554, 0.9756, 0.9085, 0.9832, 0.9777, 
        0.9660, 0.9589, 0.9531, 0.9539, 0.9755, 0.9661
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

ylim = (90, 100)

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
png_file_path = 'model_performance_metrics_v6.png'
svg_file_path = 'model_performance_metrics_v6.svg'
plt.savefig(png_file_path, format='png', dpi=300)
plt.savefig(svg_file_path, format='svg', dpi=300)

# Show plot
plt.show()
