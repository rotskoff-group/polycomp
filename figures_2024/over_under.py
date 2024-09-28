#This file calculates and plots the error for a number of results, but requires a large quantity of raw data files from timing_data

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import basename
import matplotlib.cm as cm
import os

# Define file patterns
patterns = ["*_numeric.npy", "*_neural.npy"]

# Initialize lists to store data
data = []

# Helper function to calculate the number of segments
def calculate_segments(name):
    segments = 7 * len(name)
    if name != name[::-1]:  # Check if the name is not a palindrome
        segments *= 2
    return segments

# Load data from files
for pattern in patterns:
    for file in glob(pattern):
        structure = basename(file).split('_')[0]
        data_type = basename(file).split('_')[1].split('.')[0]
        times = np.load(file)

        avg_time = np.mean(times)
        std_dev = np.std(times)
        segments = calculate_segments(structure)

        data.append([structure, data_type, segments, avg_time, std_dev])

# Convert data to numpy array
data_array = np.array(data, dtype=object)

# Save processed data to .npy file
np.save("processed_data.npy", data_array)

# Load the array for the violin plot
array = np.load('/scratch/users/epert/ABC_neural/production_ABC/process/merged_array.npy')

# Extract the middle 6 columns (2nd to 7th) and the last column
middle_columns = array[:, 1:7]  # Indices 1 to 6 (0-based)
last_column = array[:, -2]  # Last column

# Determine which column (from 1 to 6) has the maximum value for each row
dominant_columns = np.argmax(middle_columns, axis=1)

# Initialize lists to hold data for each column
data = [[] for _ in range(6)]

# Populate the data lists based on the dominant column
for i, dominant in enumerate(dominant_columns):
    data[dominant].append(last_column[i])

# Add an initial list for the whole distribution
all_data = [last_column.tolist()] + data

# Filter out empty data lists and corresponding labels
labels = [
    r'$\textrm{Overall}$', r'$\textrm{ABC}$', r'$\textrm{ABCCBA}$', r'$\textrm{CAABBC}$', r'$\textrm{AB}$', r'$\textrm{AC}$', r'$\textrm{BC}$'
]
filtered_data = [d for d in all_data if len(d) > 0]
filtered_labels = [label for d, label in zip(all_data, labels) if len(d) > 0]

# Define custom colors with transparency
colors = cm.get_cmap('tab10', len(filtered_data))
violin_colors = [colors(i) for i in range(len(filtered_data))]

# Apply the style file
stylefile_path = os.path.expanduser('~/polycomp/figures_2024/stylefile.mplstyle')
plt.style.use(stylefile_path)

# Create the figure and axes for the over-under plot with aspect ratio 2:1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18), sharex=False)

# Plotting the scatter plot and trend lines on the top subplot
for data_type, color, label in zip(["numeric", "neural"], ["blue", "red"], [r"$\textrm{Pseudospectral MDE}$", r"$\textrm{Neural\ Operator}$"]):
    subset = np.array([row for row in data_array if row[1] == data_type], dtype=object)
    segments = subset[:, 2].astype(float)
    avg_times = subset[:, 3].astype(float)
    std_devs = subset[:, 4].astype(float)

    # Plot only the data points without error bars
    ax1.plot(segments, avg_times, 'D', label=label, color=color, ms=17)

    # Trend line
    z = np.polyfit(segments, avg_times, 1)
    p = np.poly1d(z)

    # Extend the trend line to intersect the Y-axis
    extended_segments = np.linspace(0, max(segments), 100)
    ax1.plot(extended_segments, p(extended_segments), color=color, linestyle='--', lw=7)

# Configure the top subplot
ax1.set_xlabel(r"$\textrm{Number\ of\ Segments\ Integrated}$", fontsize=30)
ax1.set_ylabel(r"$\textrm{Wall Clock Time (seconds)}$", fontsize=30)
ax1.tick_params(axis='both', labelsize=28)
ax1.legend(fontsize=30)
# Add label (b)
ax1.text(-0.1, 1.1, r"$\textrm{(b)}$", transform=ax1.transAxes, fontsize=36, fontweight='bold', va='top', ha='right')

# Plotting the violin plot on the bottom subplot
parts = ax2.violinplot(filtered_data, showmeans=True, showmedians=True, showextrema=True)

# Set the color of the means, medians, and extremes to black
for partname in ['cmeans', 'cmedians', 'cmins', 'cmaxes']:
    if partname in parts:
        line_collection = parts[partname]
        line_collection.set_edgecolor('black')

# Set the color of the horizontal line (through the violin) to black
if 'cbars' in parts:
    bar_collection = parts['cbars']
    bar_collection.set_edgecolor('black')

# Set the color of the violins with transparency
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(violin_colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.6)  # Adjust transparency

# Configure the bottom subplot
ax2.set_xlabel(r"$\textrm{Dominant\ Component}$", fontsize=30)
ax2.set_ylabel(r"$\textrm{Average Relative Error}$", fontsize=30)
ax2.set_xticks(np.arange(1, len(filtered_labels) + 1))
ax2.set_xticklabels(filtered_labels, rotation=45, fontsize=26)
ax2.tick_params(axis='both', labelsize=28)
# Add label (c)
ax2.text(-0.1, 1.1, r"$\textrm{(c)}$", transform=ax2.transAxes, fontsize=36, fontweight='bold', va='top', ha='right')

# Ensure the violin plot x-axis is not aligned with the scatter plot
ax2.set_xlim(0.5, len(filtered_labels) + 0.5)

# Adjust layout to prevent overlap and ensure correct positioning
plt.subplots_adjust(hspace=0.3)  # Increase the space between subplots

# Save the plot as a PDF with twice as tall as wide aspect ratio
plt.savefig("combined_plot.pdf", bbox_inches='tight')

# Show the plot
# plt.show()

