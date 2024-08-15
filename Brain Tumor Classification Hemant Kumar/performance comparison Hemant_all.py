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
    92.9853, 95.5954, 94.6166, 84.1762, 88.9070, 90.0489, 86.7863, 95.9217, 86.1338, 93.8010, 93.8010, 92.1697
]

precision = [
    91.86, 95.82, 94.95, 82.73, 87.52, 90.52, 85.84, 95.94, 85.86, 93.69, 94.33, 91.52
]

recall = [
    93.02, 94.12, 93.35, 82.85, 87.93, 88.52, 84.23, 95.08, 88.13, 92.53, 91.82, 90.99
]

f1_score = [
    92.35, 94.77, 94.06, 81.97, 87.68, 89.12, 84.80, 95.48, 85.72, 93.06, 92.77, 91.18
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
plt.savefig("/home/saiful/brain tumor/Brain Tumor Classification Hemant Kumar/results/model_metrics_comparison_hemant.png", dpi=600)
plt.savefig("/home/saiful/brain tumor/Brain Tumor Classification Hemant Kumar/results/model_metrics_comparison_hemant.svg", dpi=600)
plt.show()
