import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the results from the simulation
results = np.load('simulation_results.npz')
solution = results['solution']
t = results['t']

# Create figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(20, 20))

# Initialize plots
line1a, = axs[0].plot([], [], 'b-', label='Total Economy (H+F)')
line1b, = axs[0].plot([], [], 'g-', label='Environmental Impact')
line2a, = axs[1].plot([], [], 'b-', label='Stock Market Index')
line2b, = axs[1].plot([], [], 'r-', label='Inflation Rate')
line2c, = axs[1].plot([], [], 'g-', label='Interest Rate')

# Set the axes labels, titles, etc.
axs[0].set_title('Economic Growth vs Environmental Impact')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Value')
axs[0].legend()

axs[1].set_title('Stock Market vs Economic Indicators')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Value / Rate')
axs[1].legend()

# Set axis limits
for ax in axs:
    ax.set_xlim(0, t[-1])

axs[0].set_ylim(min((solution[0] + solution[1]).min(), solution[10].min()),
                max((solution[0] + solution[1]).max(), solution[10].max()))
axs[1].set_ylim(min(solution[11].min(), solution[7].min(), solution[8].min()),
                max(solution[11].max(), solution[7].max(), solution[8].max()))

# Animation update function
def update(frame):
    line1a.set_data(t[:frame], solution[0, :frame] + solution[1, :frame])
    line1b.set_data(t[:frame], solution[10, :frame])
    line2a.set_data(t[:frame], solution[11, :frame])
    line2b.set_data(t[:frame], solution[7, :frame])
    line2c.set_data(t[:frame], solution[8, :frame])
    return line1a, line1b, line2a, line2b, line2c

# Create the animation
anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

plt.tight_layout()
plt.show()

# Optionally, save the animation
# anim.save('environmental_stock_market_animation.mp4', writer='ffmpeg', fps=30)