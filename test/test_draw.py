import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set up the style
plt.style.use('default')
sns.set_palette("husl")

# Create sample data that matches your distribution
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([0.15, 0.35, 0.45, 0.42, 0.38, 0.28, 0.20, 0.15, 0.08])

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the line
ax.plot(x, y, 'k-', linewidth=2, zorder=3)

# Add data points
ax.scatter(x, y, c='black', s=50, zorder=4)

# Add percentage labels at key points
labels = ['13.7%', '44.6%', '25.26%', '13.4%']
label_positions = [(1.5, 0.25), (3.5, 0.47), (5.5, 0.35), (7.5, 0.18)]

for label, pos in zip(labels, label_positions):
    ax.annotate(label, pos, fontsize=12, ha='center', va='bottom', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Add colored rectangles to represent different categories
rect_colors = ['#8B4513', '#D2691E', '#CD853F', '#A0522D']  # Brown variations
rect_positions = [(1, 0.02), (3, 0.02), (5, 0.02), (7, 0.02)]
rect_widths = [1, 1, 1, 1]
rect_heights = [0.08, 0.08, 0.08, 0.08]

for i, (pos, width, height, color) in enumerate(zip(rect_positions, rect_widths, rect_heights, rect_colors)):
    rect = Rectangle(pos, width, height, facecolor=color, edgecolor='black', linewidth=1, zorder=2)
    ax.add_patch(rect)

# Customize the grid
ax.grid(True, which='major', linestyle='-', linewidth=0.8, zorder=1)
ax.set_axisbelow(True)

# Set grid colors - gray for top and right, black for bottom and left
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')

# Set grid line colors
ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)

# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=12, colors='black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

# Set labels and title
ax.set_xlabel('PSNR difference (dB)', fontsize=14, color='black', fontweight='bold')
ax.set_ylabel('PDF', fontsize=14, color='black', fontweight='bold')

# Set axis limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 0.5)

# Set ticks
ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks(np.arange(0, 0.6, 0.1))

# Make the plot look more professional
plt.tight_layout()

# Show the plot
plt.show()

# Alternative approach with more precise grid control
def create_custom_grid_plot():
    """
    Alternative function with more precise control over grid appearance
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sample data
    x = np.linspace(0, 10, 100)
    y = 0.4 * np.exp(-((x-3)**2)/2) + 0.3 * np.exp(-((x-5.5)**2)/1.5) + 0.1 * np.exp(-((x-1.5)**2)/0.5)
    
    # Plot the curve
    ax.plot(x, y, 'k-', linewidth=2.5, zorder=3)
    
    # Remove default spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add custom spines
    # Bottom spine (black)
    ax.axhline(y=0, color='black', linewidth=1.5, zorder=2)
    # Left spine (black)  
    ax.axvline(x=0, color='black', linewidth=1.5, zorder=2)
    # Top spine (gray)
    ax.axhline(y=0.5, color='gray', linewidth=1, zorder=1)
    # Right spine (gray)
    ax.axvline(x=10, color='gray', linewidth=1, zorder=1)
    
    # Add custom grid
    for i in range(1, 10):
        ax.axvline(x=i, color='lightgray', linewidth=0.5, alpha=0.7, zorder=0)
    for i in np.arange(0.1, 0.5, 0.1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.7, zorder=0)
    
    # Customize labels and ticks
    ax.set_xlabel('PSNR difference (dB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('PDF', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.5)
    
    # Set custom tick positions
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    
    # Style the ticks
    ax.tick_params(axis='both', which='major', labelsize=12, colors='black')
    ax.tick_params(bottom=True, left=True, top=False, right=False)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("graph/test.pdf", dpi=2400)

# Call the alternative function
print("Creating alternative version with more precise grid control:")
create_custom_grid_plot()