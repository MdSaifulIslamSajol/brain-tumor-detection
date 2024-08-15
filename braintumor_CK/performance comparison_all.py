import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

# Data for the metrics of different models
models = [
    "Swinv2ForImageClassification",
    "ViTForImageClassification",
    "ConvNextV2ForImageClassification",
    "CvtForImageClassification",
    "EfficientFormerForImageClassification",
    "PvtV2ForImageClassification",
    "MobileViTV2ForImageClassification",
    "resnet50ForImageClassification",
    "vgg16ForImageClassification",
    "mobilenetForImageClassification",
    "googlenetForImageClassification",
    "efficientnet_b0ForImageClassification"
]

test_accuracy = [
    96.2237,
    96.5868,
    97.5309,
    97.6761,
    97.6761,
    95.3522,
    98.0392,
    95.7879,
    96.2237,
    95.6427,
    95.7153,
    97.9666
]

precision = [
    96.25,
    96.31,
    97.31,
    97.41,
    97.46,
    95.10,
    97.93,
    95.51,
    95.94,
    95.28,
    95.55,
    97.77
]

recall = [
    95.70,
    96.36,
    97.47,
    97.72,
    97.72,
    95.60,
    97.90,
    96.10,
    96.30,
    95.57,
    95.61,
    97.97
]

f1_score = [
    95.92,
    96.33,
    97.38,
    97.54,
    97.56,
    95.17,
    97.91,
    95.59,
    96.08,
    95.39,
    95.53,
    97.86
]

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

# Sort the models by the metrics in decreasing order
indices = np.argsort(test_accuracy)[::-1]
sorted_models = [models[i] for i in indices]
sorted_test_accuracy = [test_accuracy[i] for i in indices]
sorted_precision = [precision[i] for i in indices]
sorted_recall = [recall[i] for i in indices]
sorted_f1_score = [f1_score[i] for i in indices]

# Sort the short names according to the sorted indices
sorted_short_names = [short_names[i] for i in indices]

# Sort the colors according to the sorted indices
sorted_colors = [color_mapping[short_names[i]] for i in indices]

# Plotting with original colors for each model
fig, axs = plt.subplots(2, 2, figsize=(20, 10))

# Set common y-axis limits
ylim = (75, 100)

# Test Accuracy
axs[0, 0].bar(sorted_short_names, sorted_test_accuracy, color=sorted_colors)
axs[0, 0].set_title('Test Accuracy')
axs[0, 0].set_ylim(ylim)
axs[0, 0].set_xticklabels(sorted_short_names, rotation=45)

# Precision
axs[0, 1].bar(sorted_short_names, sorted_precision, color=sorted_colors)
axs[0, 1].set_title('Precision')
axs[0, 1].set_ylim(ylim)
axs[0, 1].set_xticklabels(sorted_short_names, rotation=45)

# Recall
axs[1, 0].bar(sorted_short_names, sorted_recall, color=sorted_colors)
axs[1, 0].set_title('Recall')
axs[1, 0].set_ylim(ylim)
axs[1, 0].set_xticklabels(sorted_short_names, rotation=45)

# F1 Score
axs[1, 1].bar(sorted_short_names, sorted_f1_score, color=sorted_colors)
axs[1, 1].set_title('F1 Score')
axs[1, 1].set_ylim(ylim)
axs[1, 1].set_xticklabels(sorted_short_names, rotation=45)

plt.tight_layout()
plt.savefig("/home/saiful/brain tumor/braintumor_CK/results/model_metrics_comparison_robocom.png", dpi=600)
plt.savefig("/home/saiful/brain tumor/braintumor_CK/results/model_metrics_comparison_robocom.svg", dpi=600)
plt.show()
