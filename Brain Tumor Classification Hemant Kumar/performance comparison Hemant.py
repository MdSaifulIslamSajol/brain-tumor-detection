import matplotlib.pyplot as plt
import numpy as np

# Data for the new scenario
models_new = [
    'Swinv2', 'ViT', 'ConvNextV2', 'Cvt', 'EfficientFormer', 'PvtV2', 
    'MobileViT', 'resnet50', 'vgg16', 'mobilenet', 'googlenet', 'efficientnet_b0'
]

train_accuracies_new = [80.0898, 86.4545, 83.4761, 75.0306, 79.5594, 83.9249, 76.7034, 88.1681, 78.0906, 85.3937, 82.7825, 80.5794]
test_accuracies_new = [94.6166, 96.2480, 94.4535, 87.1126, 88.5808, 93.6378, 88.9070, 93.4747, 89.0701, 91.6803, 92.6591, 92.1697]

# Sort models based on test accuracy in decreasing order for the new data
sorted_indices_new = np.argsort(test_accuracies_new)[::-1]
models_new = np.array(models_new)[sorted_indices_new]
train_accuracies_new = np.array(train_accuracies_new)[sorted_indices_new]
test_accuracies_new = np.array(test_accuracies_new)[sorted_indices_new]

# Plotting for the new data
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Bar colors
colors = plt.cm.tab20(np.linspace(0, 1, len(models_new)))

# Training Accuracy
ax[0].bar(models_new, train_accuracies_new, color=colors)
ax[0].set_title('Training Accuracy at Epoch 1')
ax[0].set_ylabel('Accuracy (%)')
ax[0].set_ylim(70, 100)
ax[0].set_xticklabels(models_new, rotation=45, ha='right')

# Test Accuracy
ax[1].bar(models_new, test_accuracies_new, color=colors)
ax[1].set_title('Test Accuracy at Epoch 1')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_ylim(70, 100)
ax[1].set_xticklabels(models_new, rotation=45, ha='right')

plt.tight_layout()

# Save the figure with 600 DPI
png_path_high_dpi_new = "/home/saiful/brain tumor/Brain Tumor Classification Hemant Kumar/results/model_accuracies_high_dpi_new.png"
svg_path_high_dpi_new = "/home/saiful/brain tumor/Brain Tumor Classification Hemant Kumar/results/model_accuracies_high_dpi_new.svg"
plt.savefig(png_path_high_dpi_new, dpi=600)
plt.savefig(svg_path_high_dpi_new, dpi=600)
plt.show()

png_path_high_dpi_new, svg_path_high_dpi_new
