import matplotlib.pyplot as plt

# Short names for models and their inference times
models = [
    "Swinv2", "ViT", "ConvNextV2", "Cvt", "EfficientFormer", 
    "PvtV2", "MobileViTV2", "ResNet50", "VGG16", "MobileNet", 
    "GoogleNet", "EfficientNetB0"
]
inference_times = [
    0.002807, 0.001206, 0.000676, 0.001835, 0.000525, 
    0.000834, 0.000977, 0.000697, 0.000111, 0.000348, 
    0.000586, 0.000478
]

# Colors assigned to models
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

# Sorting the order of inference times from highest to lowest
sorted_indices = sorted(range(len(inference_times)), key=lambda k: inference_times[k], reverse=True)
sorted_models = [models[i] for i in sorted_indices]
sorted_times = [inference_times[i] for i in sorted_indices]
sorted_colors = [color_map[models[i]] for i in sorted_indices]

# Convert seconds to microseconds
sorted_times_microseconds = [time * 1e6 for time in sorted_times]

# Reversing the order for plotting (highest time at top)
reversed_models = sorted_models[::-1]
reversed_times_microseconds = sorted_times_microseconds[::-1]
reversed_colors = sorted_colors[::-1]

# Plotting the data with specified requirements, font size 12, and decreasing other fonts by 2
fig, ax = plt.subplots(figsize=(7, 3.92))  # Adjusted to 16:9 ratio for IEEE paper
bars = ax.barh(reversed_models, reversed_times_microseconds, color=reversed_colors)

# Adding labels and title
ax.set_xlabel("Inference Time (microseconds)", fontsize=12)
ax.set_ylabel("Models", fontsize=12)
ax.set_title("Inference Time per Image for Different Models", fontsize=12)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='x', labelsize=10)

# Adding the inference values on top of each bar
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.0f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),  # 5 points offset to the right
                textcoords="offset points",
                ha='left', va='center', fontsize=10)

# Removing the grid
ax.grid(False)

# Adding legend with assigned colors
legend_labels = [f'{model}' for model in reversed_models]
handles = [plt.Rectangle((0,0),1,1, color=color) for color in reversed_colors]
ax.legend(handles, legend_labels, title="Models", title_fontsize=12, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# Saving the plot in SVG and PNG formats with 300 dpi
svg_path = "/mnt/data/inference_times_plot_reversed_fontsize12_ieee_16_9_colormap_reversed.svg"
png_path = "/mnt/data/inference_times_plot_reversed_fontsize12_ieee_16_9_colormap_reversed.png"

plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

# Displaying the plot
plt.show()

# Displaying paths
(svg_path, png_path)
