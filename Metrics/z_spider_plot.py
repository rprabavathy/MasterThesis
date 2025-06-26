import numpy as np
import matplotlib.pyplot as plt
import csv

# Set all text to bold
plt.rcParams['font.weight'] = 'bold'

def read_data(filename, delimiter='\t'):
    """Reads data from a CSV file. First column is labels, the rest are values."""
    labels = []
    values = []
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            metric = row[0].strip('"').strip()
            try:
                vals = list(map(float, row[1:]))
                labels.append(metric)
                values.append(vals)
            except ValueError as e:
                print(f"Skipping line due to error: {row} -> {e}")
    # Transpose so each method is a row
    return labels, np.array(values).T

def plot_radar_filled_colors(labels, data, methods, title, ax, show_legend=False):
    """Plots a radar chart with lines (no fill)."""
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Connect back to start

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=13, color='black')
    ax.set_rlabel_position(30)

    # Find min and max of data (excluding NaNs if any)
    ymin = np.nanmin(data)
    ymax = np.nanmax(data)
    # Add a little padding for clarity
    ymin = max(0, ymin - 0.02)
    ymax = min(1.0, ymax + 0.02) if ymax < 1.0 else ymax + 0.01

    # Set yticks and ylim dynamically
    ax.set_ylim(ymin, ymax)
    # Choose ticks: try to have 4-5 ticks for readability
    step = (ymax - ymin) / 4
    yticks = [round(ymin + i*step, 2) for i in range(5)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks], color="black", size=11)

    colors = ['b', 'g', 'r', 'm']  # Adjust if more methods

    for i in range(data.shape[0]):
        values = data[i, :].tolist()
        values += values[:1]  # Connect back to start
        ax.plot(angles, values, linewidth=2,  linestyle='-.', label=methods[i], color=colors[i % len(colors)])
        # DO NOT FILL: ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.15)

    if show_legend:
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), frameon=False, prop={'weight': 'bold', 'size': 15})

    # Set title color to dark blue
    ax.set_title(title, size=18, y=1.1, color='#000080', fontweight='bold')



# Define method names (adjust if needed)
methods = ['Classical : FCM with Morph', 'Classical : Kmeans', 'DL : SW', 'DL :TFRS']

# Files to plot (adjust filenames if needed)
files = ['metrics.dat', 'BG.dat', 'GM.dat', 'WM.dat']
titles = ['Overall', 'Background', 'Gray Matter', 'White Matter']

# Create a 2-row, 3-column grid (top row: 1 plot, bottom row: 3 plots)
fig = plt.figure(figsize=(14, 10))  # Larger figure for better quality

# Top row: Overall plot (spanning all columns)
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3, polar=True)
labels, data = read_data(files[0], delimiter='\t')  # Adjust delimiter if needed
plot_radar_filled_colors(labels, data, methods, titles[0], ax1, show_legend=True)

# Bottom row: Three plots (background, gray matter, white matter)
for i in range(1, 4):
    ax = plt.subplot2grid((2, 3), (1, i-1), polar=True)
    labels, data = read_data(files[i], delimiter='\t')  # Adjust delimiter if needed
    plot_radar_filled_colors(labels, data, methods, titles[i], ax)

# Adjust vertical spacing between rows
plt.subplots_adjust(hspace=0.4)

plt.tight_layout()
plt.savefig('z_radar_grid.eps', format='eps', dpi=2400, bbox_inches='tight')
plt.savefig('z_radar_grid.pdf', format='pdf', dpi=2400, bbox_inches='tight')
plt.show()

