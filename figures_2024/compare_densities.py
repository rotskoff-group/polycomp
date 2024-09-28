import matplotlib.pyplot as plt
from matplotlib import colors
import cupy as cp
import numpy as np
import warnings
import os

# Apply the style file
stylefile_path = os.path.expanduser('~/polycomp/figures_2024/stylefile.mplstyle')
plt.style.use(stylefile_path)

# Suppress warnings about too many open figures
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Define the list of seed numbers
#seed_numbers = ['3374', '1953', '1448']
#seed_numbers = ['0409', '3536', '4411']
seed_numbers = ['4088', '0175', '3437']

# Define colors for the imshow
colors_list = ["salmon", "wheat", "skyblue"]  # Red for A, Green for B, Blue for C
cmap = colors.ListedColormap(colors_list)
pie_colors = ['lightcoral', 'gold', 'lightskyblue', 'lightgreen', 'violet', 'orange']

# Create a figure with subplots (2x3 layout)
fig, axs = plt.subplots(2, len(seed_numbers), figsize=(12, 8), gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
#fig.subplots_adjust(top=0.8)

# Define the ticks for the colorbar
ticks = [0.167, 0.5, 0.833]  # Adjusted to place 'A' at 1/6 of the color range

# Labels for the subplots
subplot_labels = [r"$\textrm{(a)}$", r"$\textrm{(b)}$", r"$\textrm{(c)}$", r"$\textrm{(d)}$", r"$\textrm{(e)}$", r"$\textrm{(f)}$"]

# Loop through each seed number to generate plots
for i, seed_number in enumerate(seed_numbers):
    # Define file paths based on the seed number
    exact_image_path = f"/scratch/users/epert/ABC_neural/production_ABC/exact_dynamics/seed_{seed_number}/dens_traj.npy"
    approx_image_path = f"/scratch/users/epert/ABC_neural/production_ABC/runs_abs/seed_{seed_number}/dens_traj.npy"

    # Load the data
    exact_data = np.load(exact_image_path)
    approx_data = np.load(approx_image_path)

    # Generate random fractions based on the seed
    cp.random.seed(int(seed_number))  # Seed for reproducibility
    rand_list = cp.sort(cp.random.rand(5)).get()

    # Define the normalized amounts for the bar
    norm_amt = {
        r'$\textrm{ABC}$': rand_list[0],
        r'$\textrm{ABCCBA}$': rand_list[1] - rand_list[0],
        r'$\textrm{CAABBC}$': rand_list[2] - rand_list[1],
        r'$\textrm{AB}$': rand_list[3] - rand_list[2],
        r'$\textrm{AC}$': rand_list[4] - rand_list[3],
        r'$\textrm{BC}$': 1 - rand_list[4],
    }

    # Use the last frame of the trajectory
    last_frame_approx = np.argmax(approx_data[-1], axis=0)
    last_frame_exact = np.argmax(exact_data[-1], axis=0)

    # Plot approximation (top row)
    ax_approx = axs[0, i]
    im_anim_approx = ax_approx.imshow(last_frame_approx, cmap=cmap, vmin=0, vmax=2)
    ax_approx.set_xticks([])
    ax_approx.set_yticks([])

    # Plot exact solution (bottom row)
    ax_exact = axs[1, i]
    im_anim_exact = ax_exact.imshow(last_frame_exact, cmap=cmap, vmin=0, vmax=2)
    ax_exact.set_xticks([])
    ax_exact.set_yticks([])

    # Add labels to the subplots
    ax_approx.text(0.17, 0.97, subplot_labels[2*i], transform=ax_approx.transAxes, fontsize=28, fontweight='bold', va='top', ha='right')
    ax_exact.text(0.17, 0.97, subplot_labels[2*i + 1], transform=ax_exact.transAxes, fontsize=28, fontweight='bold', va='top', ha='right')

    # Add proportional horizontal bar directly above each pair of plots
    bar_ax = fig.add_axes([ax_approx.get_position().x0, ax_approx.get_position().y1 + 0.01, ax_approx.get_position().width + 0.012, 0.02])
    left = 0
    label_positions = []  # Track the positions of the labels
    for j, (key, value) in enumerate(norm_amt.items()):
        bar_ax.barh(0, value, left=left, color=pie_colors[j])
        if value >= 0.12:  # Only add label if the fraction is 10% or more
            label_position = left + value / 2
            bar_ax.text(
                label_position - 0.03, 0.3, key,
                ha='left',  # Horizontal alignment: center
                va='bottom',  # Vertical alignment: bottom (so the top of the text is at label_position)
                fontsize=16,
                rotation=30,  # Rotate text by 30 degrees
                color='black'
            )
            label_positions.append(label_position)  # Store the label position
        left += value
    bar_ax.set_yticks([])
    bar_ax.set_xticks([])
    bar_ax.set_frame_on(False)

# Add a single colorbar across both plots
cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.76])
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, ticks=ticks)
cbar.ax.set_yticklabels([r'$\textrm{A}$', r'$\textrm{B}$', r'$\textrm{C}$'])
cbar.ax.tick_params(labelsize=28)
cbar.ax.invert_yaxis()  # Invert the colorbar to match the cmap ordering

# Save the final plot as a PDF
pdf_filename = "comparison_all_seeds.pdf"
fig.savefig(pdf_filename, bbox_inches='tight', pad_inches=0)
# Convert seed numbers list to a string
seed_str = "_".join(seed_numbers)

# Incorporate into filename
pdf_filename = f"random_plots_{seed_str}.pdf"

# Save the final plot with the new filename
fig.savefig(pdf_filename, bbox_inches='tight', pad_inches=0)
plt.close()

plt.close()

