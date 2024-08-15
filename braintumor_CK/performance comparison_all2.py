# Plotting with Paired colors for each model without grids and with font size 12
fig, axs = plt.subplots(2, 2, figsize=(20, 10))

# Set common y-axis limits
ylim = (75, 100)

# Test Accuracy
axs[0, 0].bar(sorted_short_names, sorted_test_accuracy, color=sorted_colors_paired)
axs[0, 0].set_title('Test Accuracy', fontsize=12)
axs[0, 0].set_ylim(ylim)
axs[0, 0].set_xticklabels(sorted_short_names, rotation=45, fontsize=12)
axs[0, 0].tick_params(axis='y', labelsize=12)
axs[0, 0].grid(False)

# Precision
axs[0, 1].bar(sorted_short_names, sorted_precision, color=sorted_colors_paired)
axs[0, 1].set_title('Precision', fontsize=12)
axs[0, 1].set_ylim(ylim)
axs[0, 1].set_xticklabels(sorted_short_names, rotation=45, fontsize=12)
axs[0, 1].tick_params(axis='y', labelsize=12)
axs[0, 1].grid(False)

# Recall
axs[1, 0].bar(sorted_short_names, sorted_recall, color=sorted_colors_paired)
axs[1, 0].set_title('Recall', fontsize=12)
axs[1, 0].set_ylim(ylim)
axs[1, 0].set_xticklabels(sorted_short_names, rotation=45, fontsize=12)
axs[1, 0].tick_params(axis='y', labelsize=12)
axs[1, 0].grid(False)

# F1 Score
axs[1, 1].bar(sorted_short_names, sorted_f1_score, color=sorted_colors_paired)
axs[1, 1].set_title('F1 Score', fontsize=12)
axs[1, 1].set_ylim(ylim)
axs[1, 1].set_xticklabels(sorted_short_names, rotation=45, fontsize=12)
axs[1, 1].tick_params(axis='y', labelsize=12)
axs[1, 1].grid(False)

plt.tight_layout()

# Save the plot with specified DPI settings
fig.savefig("/mnt/data/model_metrics_comparison_paired_fontsize12_300dpi.png", dpi=300)
fig.savefig("/mnt/data/model_metrics_comparison_paired_fontsize12_600dpi.svg", dpi=600)

# Display the plot
plt.show()
