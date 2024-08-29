import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

# Load the results from the simulation
results = np.load('simulation_results.npz')
solution = results['solution']
t = results['t']

# Create figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

# Create a normalized colormap
norm = Normalize(vmin=t.min(), vmax=t.max())

# Initialize scatter plots
scatters = []
for ax in axs.flatten():
    scatters.append(ax.scatter([], [], c=[], cmap='viridis', norm=norm))

# Set the axes labels, titles, etc.
axs[0, 0].set_xlabel('Household Wealth')
axs[0, 0].set_ylabel('Firm Value')
axs[0, 0].set_title('Phase Portrait: Household Wealth vs Firm Value')

axs[0, 1].set_xlabel('Unemployment Rate')
axs[0, 1].set_ylabel('Inflation Rate')
axs[0, 1].set_title('Phase Portrait: Unemployment vs Inflation (Phillips Curve)')

axs[1, 0].set_xlabel('Interest Rate')
axs[1, 0].set_ylabel('Inflation Rate')
axs[1, 0].set_title('Phase Portrait: Interest Rate vs Inflation')

axs[1, 1].set_xlabel('Environmental Impact')
axs[1, 1].set_ylabel('Total Economy')
axs[1, 1].set_title('Phase Portrait: Environmental Impact vs Total Economy')

# Set axis limits
axs[0, 0].set_xlim(solution[0].min(), solution[0].max())
axs[0, 0].set_ylim(solution[1].min(), solution[1].max())

axs[0, 1].set_xlim(solution[6].min(), solution[6].max())
axs[0, 1].set_ylim(solution[7].min(), solution[7].max())

axs[1, 0].set_xlim(solution[8].min(), solution[8].max())
axs[1, 0].set_ylim(solution[7].min(), solution[7].max())

total_economy = solution[0] + solution[1]
axs[1, 1].set_xlim(solution[10].min(), solution[10].max())
axs[1, 1].set_ylim(total_economy.min(), total_economy.max())

# Add colorbars
for ax, scatter in zip(axs.flatten(), scatters):
    plt.colorbar(scatter, ax=ax, label='Time')

# Animation update function
def update(frame):
    scatters[0].set_offsets(np.c_[solution[0, :frame], solution[1, :frame]])
    scatters[1].set_offsets(np.c_[solution[6, :frame], solution[7, :frame]])
    scatters[2].set_offsets(np.c_[solution[8, :frame], solution[7, :frame]])
    total_economy = solution[0, :frame] + solution[1, :frame]
    scatters[3].set_offsets(np.c_[solution[10, :frame], total_economy])
    
    for scatter in scatters:
        scatter.set_array(t[:frame])
    
    return scatters

# Create the animation
anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

plt.tight_layout()

# To avoid the AttributeError, use plt.show() in interactive mode
plt.ion()
plt.show()

# Keep the plot open
plt.ioff()
plt.show()

# Optionally, save the animation
# anim.save('phase_portraits_animation.mp4', writer='ffmpeg', fps=30)