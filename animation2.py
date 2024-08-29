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
fig, axs = plt.subplots(2, 1, figsize=(20, 20))

# Create a normalized colormap
norm = Normalize(vmin=t.min(), vmax=t.max())

# Initialize plots
lines1 = [axs[0].plot([], [], label=label)[0] for label in ['Number of Households', 'Number of Firms']]
scatter2 = axs[1].scatter([], [], c=[], cmap='viridis', norm=norm)
lines3 = [axs[1].plot([], [], label=label, color=color)[0] for label, color in 
          [('Number of Households', 'b'), ('Number of Firms', 'g'), ('Unemployment Rate', 'r')]]

# Set the axes labels, titles, etc.
axs[0].set_title('Population over Time')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Population')
axs[0].legend()

axs[1].set_title('Population Dynamics and Unemployment')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Population / Rate')
axs[1].legend()

# Set axis limits
for ax in axs:
    ax.set_xlim(0, t[-1])

axs[0].set_ylim(min(solution[4].min(), solution[5].min()), max(solution[4].max(), solution[5].max()))
axs[1].set_ylim(min(solution[4].min(), solution[5].min(), solution[6].min()),
                max(solution[4].max(), solution[5].max(), solution[6].max()))

# Add colorbar
plt.colorbar(scatter2, ax=axs[1], label='Time')

# Animation update function
def update(frame):
    for i, line in enumerate(lines1):
        line.set_data(t[:frame], solution[i+4, :frame])
    scatter2.set_offsets(np.c_[solution[4, :frame] + solution[5, :frame], gdp[:frame]])
    scatter2.set_array(t[:frame])
    for i, line in enumerate(lines3):
        line.set_data(t[:frame], solution[i+4, :frame])
    return tuple(lines1) + (scatter2,) + tuple(lines3)

# Create the animation
anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

plt.tight_layout()
plt.show()

# Optionally, save the animation
# anim.save('population_unemployment_animation.mp4', writer='ffmpeg', fps=30)