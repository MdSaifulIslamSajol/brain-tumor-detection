import matplotlib.pyplot as plt
import numpy as np

# Data for the new dataset
models_new = [
    'Swinv2', 'ViT', 'ConvNextV2', 'Cvt', 'EfficientFormer', 'PvtV2', 
    'MobileViT', 'resnet50', 'vgg16', 'mobilenet', 'googlenet', 'efficientnet_b0'
]

train_accuracies_new = [81.4694, 87.3061, 87.5918, 74.0408, 83.1020, 85.2653, 75.2653, 89.7551, 80.3673, 87.4694, 84.0408, 82.5306]
test_accuracies_new = [92.9853, 95.2692, 96.4111, 92.8222, 86.2969, 93.3116, 87.1126, 95.1060, 90.3752, 91.8434, 92.3328, 87.7651]

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

png_path_high_dpi_new = "/home/saiful/brain tumor/brain tumor figshare/results/model_accuracies_high_dpi_new_dataset.png"
svg_path_high_dpi_new = "/home/saiful/brain tumor/brain tumor figshare/results/model_accuracies_high_dpi_new_dataset.svg"
plt.savefig(png_path_high_dpi_new, dpi=600)
plt.savefig(svg_path_high_dpi_new, dpi=600)
plt.show()

png_path_high_dpi_new, svg_path_high_dpi_new
