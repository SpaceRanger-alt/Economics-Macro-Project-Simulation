import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

# Load the results from the simulation
results = np.load('simulation_results.npz')
solution = results['solution']
t = results['t']
gdp = results['gdp']

# Create figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

# Create a normalized colormap
norm = Normalize(vmin=t.min(), vmax=t.max())

# Initialize plots
lines1 = [axs[0, 0].plot([], [], label=label)[0] for label in ['Households', 'Firms', 'Government', 'Central Bank']]
line2, = axs[0, 1].plot([], [])
scatter3 = axs[1, 0].scatter([], [], c=[], cmap='viridis', norm=norm)
line4a, = axs[1, 1].plot([], [], label='Total Economy (H+F)')
line4b, = axs[1, 1].plot([], [], label='Government Spending')

# Set the axes labels, titles, etc.
axs[0, 0].set_title('Time Series of State Variables')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()

axs[0, 1].set_title('GDP over Time')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('GDP')

axs[1, 0].set_title('Inflation vs Interest Rate')
axs[1, 0].set_xlabel('Inflation Rate')
axs[1, 0].set_ylabel('Interest Rate')

axs[1, 1].set_title('Normalized Economic Growth and Government Spending')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Normalized Value')
axs[1, 1].legend()

# Set axis limits
for ax in axs.flat:
    ax.set_xlim(0, t[-1])

axs[0, 0].set_ylim(min(solution[0].min(), solution[1].min(), solution[2].min(), solution[3].min()),
                   max(solution[0].max(), solution[1].max(), solution[2].max(), solution[3].max()))
axs[0, 1].set_ylim(gdp.min(), gdp.max())
axs[1, 0].set_xlim(solution[7].min(), solution[7].max())
axs[1, 0].set_ylim(solution[8].min(), solution[8].max())
axs[1, 1].set_ylim(0, 1)

# Add colorbar
plt.colorbar(scatter3, ax=axs[1, 0], label='Time')

# Animation update function
def update(frame):
    for i, line in enumerate(lines1):
        line.set_data(t[:frame], solution[i, :frame])
    line2.set_data(t[:frame], gdp[:frame])
    scatter3.set_offsets(np.c_[solution[7, :frame], solution[8, :frame]])
    scatter3.set_array(t[:frame])
    line4a.set_data(t[:frame], (solution[0, :frame] + solution[1, :frame]) / np.max(solution[0] + solution[1]))
    line4b.set_data(t[:frame], solution[2, :frame] / np.max(solution[2]))
    return tuple(lines1) + (line2, scatter3, line4a, line4b)

# Create the animation
anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

plt.tight_layout()
plt.show()

# Optionally, save the animation
# anim.save('economic_indicators_animation.mp4', writer='ffmpeg', fps=30)