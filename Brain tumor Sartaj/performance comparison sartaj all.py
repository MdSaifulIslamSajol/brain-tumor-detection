import matplotlib.pyplot as plt
import numpy as np

# Short names for models (initially sorted by the initial data)
short_names = [
    "Swinv2", "ViT", "ConvNextV2", "Cvt", "EfficientFormer", 
    "PvtV2", "MobileViTV2", "ResNet50", "VGG16", "MobileNet", 
    "GoogleNet", "EfficientNetB0"
]

# Original popular colors for each model, ensuring there are 12 distinct colors
original_colors = [
    'blue', 'green', 'red', 'cyan', 'magenta', 
    'skyblue', 'orange', 'violet', 'brown', 'pink', 'gray', 'yellow'
]

# Mapping models to their colors
color_mapping = {short_name: color for short_name, color in zip(short_names, original_colors)}

# New data for the models
new_test_accuracy = [
    70.8122,
    68.7817,
    68.5279,
    63.9594,
    58.8832,
    65.2284,
    43.9086,
    69.0355,
    53.0457,
    68.7817,
    64.2132,
    64.4670
]

new_precision = [
    76.77,
    76.54,
    76.15,
    71.24,
    69.92,
    73.36,
    51.11,
    77.68,
    55.48,
    73.02,
    69.83,
    75.99
]

new_recall = [
    69.81,
    66.48,
    65.97,
    63.09,
    56.80,
    61.67,
    46.08,
    66.88,
    53.94,
    67.58,
    63.56,
    63.14
]

new_f1_score = [
    66.70,
    64.93,
    65.75,
    62.92,
    54.48,
    59.75,
    43.18,
    64.82,
    53.91,
    66.70,
    61.00,
    61.08
]

# Function to sort and get sorted metrics, names, and colors
def sort_metrics(metric):
    indices = np.argsort(metric)[::-1]
    sorted_metric = [metric[i] for i in indices]
    sorted_short_names = [short_names[i] for i in indices]
    sorted_colors = [color_mapping[name] for name in sorted_short_names]
    return sorted_metric, sorted_short_names, sorted_colors

# Plotting with assigned colors for each model
fig, axs = plt.subplots(2, 2, figsize=(20, 10))

# Set common y-axis limits
ylim = (0, 100)

# Test Accuracy
sorted_test_accuracy, sorted_short_names, sorted_colors = sort_metrics(new_test_accuracy)
axs[0, 0].bar(sorted_short_names, sorted_test_accuracy, color=sorted_colors)
axs[0, 0].set_title('Test Accuracy')
axs[0, 0].set_ylim(ylim)
axs[0, 0].set_xticklabels(sorted_short_names, rotation=45)

# Precision
sorted_precision, sorted_short_names, sorted_colors = sort_metrics(new_precision)
axs[0, 1].bar(sorted_short_names, sorted_precision, color=sorted_colors)
axs[0, 1].set_title('Precision')
axs[0, 1].set_ylim(ylim)
axs[0, 1].set_xticklabels(sorted_short_names, rotation=45)

# Recall
sorted_recall, sorted_short_names, sorted_colors = sort_metrics(new_recall)
axs[1, 0].bar(sorted_short_names, sorted_recall, color=sorted_colors)
axs[1, 0].set_title('Recall')
axs[1, 0].set_ylim(ylim)
axs[1, 0].set_xticklabels(sorted_short_names, rotation=45)

# F1 Score
sorted_f1_score, sorted_short_names, sorted_colors = sort_metrics(new_f1_score)
axs[1, 1].bar(sorted_short_names, sorted_f1_score, color=sorted_colors)
axs[1, 1].set_title('F1 Score')
axs[1, 1].set_ylim(ylim)
axs[1, 1].set_xticklabels(sorted_short_names, rotation=45)

plt.tight_layout()
plt.savefig("/home/saiful/brain tumor/Brain tumor Sartaj/results/model_metrics_comparison_sartaj.png", dpi=600)
plt.savefig("/home/saiful/brain tumor/Brain tumor Sartaj/results/model_metrics_comparison_sartaj.svg", dpi=600)
plt.show()
