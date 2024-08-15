import pandas as pd
import matplotlib.pyplot as plt
# New data provided
data = {
    'Model': [
        'Swinv2', 'ViT', 'ConvNextV2', 
        'Cvt', 'EfficientFormer', 'PvtV2', 
        'MobileViTV2', 'ResNet50', 'VGG16', 
        'MobileNet', 'GoogleNet', 'EfficientNetB0'
    ],
    'Test Accuracy': [
        92.9853, 95.5954, 94.6166, 84.1762, 88.9070, 90.0489, 
        86.7863, 95.9217, 86.1338, 93.8010, 93.8010, 92.1697
    ],
    'Precision': [
        0.9186, 0.9582, 0.9495, 0.8273, 0.8752, 0.9052, 
        0.8584, 0.9594, 0.8586, 0.9369, 0.9433, 0.9152
    ],
    'Recall': [
        0.9302, 0.9412, 0.9335, 0.8285, 0.8793, 0.8852, 
        0.8423, 0.9508, 0.8813, 0.9253, 0.9182, 0.9099
    ],
    'F1 Score': [
        0.9235, 0.9477, 0.9406, 0.8197, 0.8768, 0.8912, 
        0.8480, 0.9548, 0.8572, 0.9306, 0.9277, 0.9118
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

ylim = (80, 100)

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
png_file_path = '/mnt/data/model_performance_metrics_v5.png'
svg_file_path = '/mnt/data/model_performance_metrics_v5.svg'
plt.savefig(png_file_path, format='png', dpi=300)
plt.savefig(svg_file_path, format='svg', dpi=300)

# Show plot
plt.show()
