
import matplotlib.pyplot as plt
import numpy as np
# Data for the new dataset
models_new = [
    'Swinv2', 'ViT', 'ConvNextV2', 'Cvt', 'EfficientFormer', 'PvtV2', 
    'MobileViT', 'resnet50', 'vgg16', 'mobilenet', 'googlenet', 'efficientnet_b0'
]

train_accuracies_new = [76.9686, 85.0871, 82.6829, 69.2683, 75.8885, 79.1289, 69.6167, 86.6899, 75.2613, 80.9408, 78.3624, 75.7143]
test_accuracies_new = [65.2284, 69.5431, 73.0964, 50.0000, 63.1980, 71.5736, 45.4315, 68.5279, 53.2995, 64.9746, 68.0203, 57.8680]

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
ax[0].set_ylim(60, 90)
ax[0].set_xticklabels(models_new, rotation=45, ha='right')

# Test Accuracy
ax[1].bar(models_new, test_accuracies_new, color=colors)
ax[1].set_title('Test Accuracy at Epoch 1')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_ylim(40, 80)
ax[1].set_xticklabels(models_new, rotation=45, ha='right')

plt.tight_layout()

# Save the figure with 600 DPI
png_path_high_dpi_new = "/home/saiful/brain tumor/Brain tumor Sartaj/results/model_accuracies_high_dpi_new_dataset_2.png"
svg_path_high_dpi_new = "/home/saiful/brain tumor/Brain tumor Sartaj/results/model_accuracies_high_dpi_new_dataset_2.svg"
plt.savefig(png_path_high_dpi_new, dpi=600)
plt.savefig(svg_path_high_dpi_new, dpi=600)
plt.show()

png_path_high_dpi_new, svg_path_high_dpi_new
