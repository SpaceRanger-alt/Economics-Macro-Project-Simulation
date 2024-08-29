import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Load the results from the simulation
results = np.load('simulation_results.npz')
solution = results['solution']
t = results['t']

# Create the main figure
fig = plt.figure(figsize=(20, 16))

# Create subplots
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')

# Initialize scatter plots with full data
scatter1 = ax1.scatter(solution[0], solution[1], solution[2], c=t, cmap='viridis')
scatter2 = ax2.scatter(solution[6], solution[7], solution[8], c=t, cmap='viridis')
scatter3 = ax3.scatter(solution[4], solution[5], solution[3], c=t, cmap='viridis')
scatter4 = ax4.scatter(solution[10], solution[9], solution[11], c=t, cmap='viridis')

# Set labels and titles
ax1.set_xlabel('Household Wealth')
ax1.set_ylabel('Firm Value')
ax1.set_zlabel('Government Spending')
ax1.set_title('Household Wealth, Firm Value, Government Spending')

ax2.set_xlabel('Unemployment Rate')
ax2.set_ylabel('Inflation Rate')
ax2.set_zlabel('Interest Rate')
ax2.set_title('Unemployment, Inflation, Interest Rate')

ax3.set_xlabel('Number of Households')
ax3.set_ylabel('Number of Firms')
ax3.set_zlabel('Central Bank Assets')
ax3.set_title('Households, Firms, Central Bank Assets')

ax4.set_xlabel('Environmental Impact')
ax4.set_ylabel('International Trade Balance')
ax4.set_zlabel('Stock Market Index')
ax4.set_title('Environment, Trade, Stock Market')

# Add colorbars
plt.colorbar(scatter1, ax=ax1, label='Time')
plt.colorbar(scatter2, ax=ax2, label='Time')
plt.colorbar(scatter3, ax=ax3, label='Time')
plt.colorbar(scatter4, ax=ax4, label='Time')

# Add a slider
slider_ax = plt.axes([0.1, 0.02, 0.8, 0.03])
slider = Slider(slider_ax, 'Time', 0, len(t)-1, valinit=len(t)-1, valstep=1)

# Update function for the slider
def update(val):
    idx = int(slider.val)
    
    scatter1._offsets3d = (solution[0, :idx+1], solution[1, :idx+1], solution[2, :idx+1])
    scatter2._offsets3d = (solution[6, :idx+1], solution[7, :idx+1], solution[8, :idx+1])
    scatter3._offsets3d = (solution[4, :idx+1], solution[5, :idx+1], solution[3, :idx+1])
    scatter4._offsets3d = (solution[10, :idx+1], solution[9, :idx+1], solution[11, :idx+1])
    
    for scatter in [scatter1, scatter2, scatter3, scatter4]:
        scatter.set_array(t[:idx+1])
    
    fig.canvas.draw_idle()

# Connect the slider to the update function
slider.on_changed(update)

# Initial call to update function
update(len(t)-1)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()