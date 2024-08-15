import matplotlib.pyplot as plt
import numpy as np

# Models and their corresponding metrics
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
    95.4323, 96.4111, 90.7015, 84.5024, 90.8646, 93.3116, 88.9070, 96.9005, 90.0489, 93.8010, 93.1485, 92.0065
]

precision = [
    95.47, 95.70, 90.34, 87.52, 90.74, 92.52, 87.76, 96.35, 89.48, 93.02, 92.92, 91.66
]

recall = [
    94.57, 96.78, 92.26, 80.40, 89.00, 93.65, 87.99, 97.06, 90.60, 93.53, 91.94, 90.78
]

f1_score = [
    94.94, 96.16, 90.49, 82.21, 89.71, 92.93, 87.85, 96.64, 89.53, 93.26, 92.20, 91.14
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
plt.savefig("/home/saiful/brain tumor/brain tumor figshare/results/model_metrics_comparison_figshare.png", dpi=600)
plt.savefig("/home/saiful/brain tumor/brain tumor figshare/results/model_metrics_comparison_figshare.svg", dpi=600)
plt.show()
