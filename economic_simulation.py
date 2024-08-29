import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') 
from scipy.integrate import solve_ivp
from numba import jit, vectorize
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

@vectorize(['float64(float64, float64, float64, float64)'])
def update_agent(value, growth_rate, volatility, dt):
    return value * np.exp((growth_rate - 0.5 * volatility**2) * dt +
                          volatility * np.sqrt(dt) * np.random.normal())

@jit(nopython=True)
def update_agents(values, growth_rates, volatilities, dt, sample_size):
    indices = np.random.choice(len(values), sample_size, replace=False)
    values[indices] = update_agent(values[indices], growth_rates[indices], volatilities[indices], dt)
    return values

@jit(nopython=True)
def advanced_economic_model(t, y, params):
    H, F, G, C, N_h, N_f, U, I, R, T, E, S = y
    (alpha, beta, gamma, zeta, mu, delta, epsilon, lambdaa, kay, tau, pi, phi, omega,
     br, dr, ir, er, nfr, brr, sig_u, sig_i, sig_r, sig_t, rho, eta, theta,
     kappa, xi, psi, chi) = params

    # Improved model equations
    dHdt = (alpha*F/N_f) - (beta*H/N_h) + (gamma*G/N_h) + (zeta*C/N_h) - (eta*U*H) + (theta*S*H/1000) - (kappa*I*H)
    dFdt = (delta*H*N_h/N_f) - (epsilon*F) + (lambdaa*F)*((1-F/N_f)/kay) - (kappa*I*F) + (xi*T/1000)
    dGdt = (tau*(H*N_h + F*N_f)/1000) - (gamma*G) + pi*(np.sin(phi*t)) - (psi*E*G/100)
    dCdt = (omega*(H*N_h + F*N_f - C)/1000) - (chi*R*C)
    dN_hdt = (br - dr)*N_h + ir - er
    dN_fdt = nfr*N_f - brr*F/1000
    dUdt = sig_u * (N_h - (H + F)/N_f)/N_h - rho*U
    dIdt = sig_i * (C/(H*N_h + F*N_f) - I)
    dRdt = sig_r * (I - R)
    dTdt = sig_t * (F/N_f - H/N_h)
    dEdt = -0.01 * E + 0.001 * (H*N_h + F*N_f)/1e6
    dSdt = 0.1 * (F/N_f) - 0.05 * S + 0.01 * (1 - U)

    return np.array([dHdt, dFdt, dGdt, dCdt, dN_hdt, dN_fdt, dUdt, dIdt, dRdt, dTdt, dEdt, dSdt])

# Adjusted parameters
params = np.array([
    0.3,    # alpha: Firm contribution to household income
    0.6,    # beta: Household spending rate
    0.35,   # gamma: Government spending rate
    0.02,   # zeta: Central bank impact on household wealth
    0.0001, # mu: Quadratic growth term for households (removed from equations)
    0.35,   # delta: Household contribution to firm revenue
    0.70,   # epsilon: Firm expenses rate
    0.08,   # lambda: Firm growth rate
    1e6,    # kay: Carrying capacity for firms
    0.25,   # tau: Tax rate
    0.05,   # pi: Amplitude of cyclical policy changes
    0.20,   # phi: Frequency of cyclical policy changes
    0.075,  # omega: Central bank adjustment rate
    0.014,  # br: Birth rate
    0.008,  # dr: Death rate
    1000,   # ir: Immigration rate (absolute number)
    500,    # er: Emigration rate (absolute number)
    0.03,   # nfr: New firm rate
    0.01,   # brr: Bankruptcy rate
    0.10,   # sig_u: Unemployment rate adjustment factor
    0.05,   # sig_i: Inflation rate adjustment factor
    0.03,   # sig_r: Interest rate adjustment factor
    0.05,   # sig_t: International trade balance adjustment factor
    0.2,    # rho: Unemployment decay rate
    0.1,    # eta: Impact of unemployment on household wealth
    0.05,   # theta: Impact of stock market on household wealth
    0.1,    # kappa: Impact of inflation on firm value and household wealth
    0.1,    # xi: Impact of international trade on firm value
    0.05,   # psi: Impact of environmental factors on government spending
    0.1     # chi: Impact of interest rates on central bank policy
])

# Adjusted initial conditions
y0 = np.array([
    100000,    # H: Total household wealth
    4000000,   # F: Total firm value
    1500000,   # G: Government spending
    5000000,   # C: Central bank assets
    1000000,   # N_h: Number of households
    200000,    # N_f: Number of firms
    0.05,      # U: Unemployment rate
    0.02,      # I: Inflation rate
    0.03,      # R: Interest rate
    1000,      # T: International trade balance
    100,       # E: Environmental impact
    1000       # S: Stock market index
])
# Initialize agents
num_households = 100000
num_firms = 20000
household_values = np.random.normal(1000, 200, num_households)
firm_values = np.random.normal(5000, 1000, num_firms)
household_growth_rates = np.full(num_households, 0.02)
firm_growth_rates = np.full(num_firms, 0.03)
household_volatilities = np.full(num_households, 0.1)
firm_volatilities = np.full(num_firms, 0.15)

# Solve the system
t_span = (0, 100)
sol = solve_ivp(advanced_economic_model, t_span, y0, args=(params,), method='RK45', dense_output=True)

# Extract solutions
num_time_points = 1000
t = np.linspace(t_span[0], t_span[1], num_time_points)
solution = sol.sol(t)

# Calculate GDP
gdp = np.zeros(num_time_points)
dt = t[1] - t[0]
household_sample_size = min(1000, num_households)
firm_sample_size = min(200, num_firms)

for i in range(num_time_points):
    household_values = update_agents(household_values, household_growth_rates, household_volatilities, dt, household_sample_size)
    firm_values = update_agents(firm_values, firm_growth_rates, firm_volatilities, dt, firm_sample_size)

    total_household_spending = np.sum(household_values) * params[1]  # beta: Household spending rate
    total_firm_production = np.sum(firm_values) * (1 - params[6])  # 1 - epsilon: Firm production rate
    gdp[i] = total_household_spending + total_firm_production

# Plotting

# Create a new figure for state space plots
fig = plt.figure(figsize=(20, 20))

# 1. Household Wealth, Firm Value, and Government Spending
ax1 = fig.add_subplot(221, projection='3d')
scatter = ax1.scatter(solution[0], solution[1], solution[2], c=t, cmap='viridis')
ax1.set_xlabel('Household Wealth')
ax1.set_ylabel('Firm Value')
ax1.set_zlabel('Government Spending')
ax1.set_title('State Space: Household Wealth, Firm Value, Government Spending')
fig.colorbar(scatter, ax=ax1, label='Time')

# 2. Unemployment, Inflation, and Interest Rate
ax2 = fig.add_subplot(222, projection='3d')
scatter = ax2.scatter(solution[6], solution[7], solution[8], c=t, cmap='viridis')
ax2.set_xlabel('Unemployment Rate')
ax2.set_ylabel('Inflation Rate')
ax2.set_zlabel('Interest Rate')
ax2.set_title('State Space: Unemployment, Inflation, Interest Rate')
fig.colorbar(scatter, ax=ax2, label='Time')

# 3. Number of Households, Number of Firms, and Central Bank Assets
ax3 = fig.add_subplot(223, projection='3d')
scatter = ax3.scatter(solution[4], solution[5], solution[3], c=t, cmap='viridis')
ax3.set_xlabel('Number of Households')
ax3.set_ylabel('Number of Firms')
ax3.set_zlabel('Central Bank Assets')
ax3.set_title('State Space: Households, Firms, Central Bank Assets')
fig.colorbar(scatter, ax=ax3, label='Time')

# 4. Environmental Impact, International Trade, and Stock Market Index
ax4 = fig.add_subplot(224, projection='3d')
scatter = ax4.scatter(solution[10], solution[9], solution[11], c=t, cmap='viridis')
ax4.set_xlabel('Environmental Impact')
ax4.set_ylabel('International Trade Balance')
ax4.set_zlabel('Stock Market Index')
ax4.set_title('State Space: Environment, Trade, Stock Market')
fig.colorbar(scatter, ax=ax4, label='Time')

plt.tight_layout()
plt.show()

# 2D phase portraits
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))

# 1. Household Wealth vs Firm Value
scatter = ax1.scatter(solution[0], solution[1], c=t, cmap='viridis')
ax1.set_xlabel('Household Wealth')
ax1.set_ylabel('Firm Value')
ax1.set_title('Phase Portrait: Household Wealth vs Firm Value')
fig.colorbar(scatter, ax=ax1, label='Time')

# 2. Unemployment vs Inflation (Phillips Curve)
scatter = ax2.scatter(solution[6], solution[7], c=t, cmap='viridis')
ax2.set_xlabel('Unemployment Rate')
ax2.set_ylabel('Inflation Rate')
ax2.set_title('Phase Portrait: Unemployment vs Inflation (Phillips Curve)')
fig.colorbar(scatter, ax=ax2, label='Time')

# 3. Interest Rate vs Inflation
scatter = ax3.scatter(solution[8], solution[7], c=t, cmap='viridis')
ax3.set_xlabel('Interest Rate')
ax3.set_ylabel('Inflation Rate')
ax3.set_title('Phase Portrait: Interest Rate vs Inflation')
fig.colorbar(scatter, ax=ax3, label='Time')

# 4. Environmental Impact vs Total Economy
total_economy = solution[0] + solution[1]  # Household Wealth + Firm Value
scatter = ax4.scatter(solution[10], total_economy, c=t, cmap='viridis')
ax4.set_xlabel('Environmental Impact')
ax4.set_ylabel('Total Economy')
ax4.set_title('Phase Portrait: Environmental Impact vs Total Economy')
fig.colorbar(scatter, ax=ax4, label='Time')

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(5, 2, figsize=(20, 50))

# 1. Time Series of State Variables
axs[0, 0].plot(t, solution[0], label='Households')
axs[0, 0].plot(t, solution[1], label='Firms')
axs[0, 0].plot(t, solution[2], label='Government')
axs[0, 0].plot(t, solution[3], label='Central Bank')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Value')
axs[0, 0].set_title('Time Series of State Variables')
axs[0, 0].legend()

# 2. GDP over Time
axs[0, 1].plot(t, gdp)
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('GDP')
axs[0, 1].set_title('GDP over Time')

# 3. Population Dynamics
axs[1, 0].plot(t, solution[4], label='Number of Households')
axs[1, 0].plot(t, solution[5], label='Number of Firms')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Population')
axs[1, 0].set_title('Population over Time')
axs[1, 0].legend()

# 4. GDP vs Population
total_population = solution[4] + solution[5]
scatter = axs[1, 1].scatter(total_population, gdp, c=t, cmap='viridis')
axs[1, 1].set_xlabel('Total Population (Households + Firms)')
axs[1, 1].set_ylabel('GDP')
axs[1, 1].set_title('GDP vs Population')
plt.colorbar(scatter, ax=axs[1, 1], label='Time')

# 5. Economic Growth and Stability (Normalized)
total_economy = solution[0] + solution[1]
axs[2, 0].plot(t, total_economy / np.max(total_economy), label='Total Economy (H+F)')
axs[2, 0].plot(t, solution[2] / np.max(solution[2]), label='Government Spending')
axs[2, 0].set_xlabel('Time')
axs[2, 0].set_ylabel('Normalized Value')
axs[2, 0].set_title('Normalized Economic Growth and Government Spending')
axs[2, 0].legend()

# 6. Population Dynamics and Unemployment
ax2 = axs[2, 1].twinx()
axs[2, 1].plot(t, solution[4], 'b-', label='Number of Households')
axs[2, 1].plot(t, solution[5], 'g-', label='Number of Firms')
ax2.plot(t, solution[6], 'r-', label='Unemployment Rate')
axs[2, 1].set_xlabel('Time')
axs[2, 1].set_ylabel('Population')
ax2.set_ylabel('Unemployment Rate')
axs[2, 1].set_title('Population Dynamics and Unemployment')
lines1, labels1 = axs[2, 1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[2, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 7. Economic Indicators Correlation
scatter = axs[3, 0].scatter(solution[7], solution[8], c=t, cmap='viridis')
axs[3, 0].set_xlabel('Inflation Rate')
axs[3, 0].set_ylabel('Interest Rate')
axs[3, 0].set_title('Inflation vs Interest Rate')
plt.colorbar(scatter, ax=axs[3, 0], label='Time')

# 8. Wealth Distribution Over Time
household_wealth = solution[0] / solution[4]
firm_wealth = solution[1] / solution[5]
axs[3, 1].plot(t, household_wealth, label='Average Household Wealth')
axs[3, 1].plot(t, firm_wealth, label='Average Firm Value')
axs[3, 1].set_xlabel('Time')
axs[3, 1].set_ylabel('Average Wealth/Value')
axs[3, 1].set_title('Wealth Distribution Over Time')
axs[3, 1].legend()

# 9. Environmental Impact vs Economic Growth
ax9 = axs[4, 0].twinx()
axs[4, 0].plot(t, total_economy, 'b-', label='Total Economy (H+F)')
ax9.plot(t, solution[10], 'g-', label='Environmental Impact')
axs[4, 0].set_xlabel('Time')
axs[4, 0].set_ylabel('Economic Value')
ax9.set_ylabel('Environmental Impact')
axs[4, 0].set_title('Economic Growth vs Environmental Impact')
lines1, labels1 = axs[4, 0].get_legend_handles_labels()
lines2, labels2 = ax9.get_legend_handles_labels()
axs[4, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 10. Stock Market Performance vs Economic Indicators
ax10 = axs[4, 1].twinx()
axs[4, 1].plot(t, solution[11], 'b-', label='Stock Market Index')
ax10.plot(t, solution[7], 'r-', label='Inflation Rate')
ax10.plot(t, solution[8], 'g-', label='Interest Rate')
axs[4, 1].set_xlabel('Time')
axs[4, 1].set_ylabel('Stock Market Index')
ax10.set_ylabel('Rate')
axs[4, 1].set_title('Stock Market vs Economic Indicators')
lines1, labels1 = axs[4, 1].get_legend_handles_labels()
lines2, labels2 = ax10.get_legend_handles_labels()
axs[4, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

# Wealth Distribution Histogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

time_points = [0, len(t)//2, -1]  # Start, middle, end
for i, tp in enumerate(time_points):
    ax1.hist(household_values, bins=50, alpha=0.5, density=True, label=f't={t[tp]:.1f}')
    ax2.hist(firm_values, bins=50, alpha=0.5, density=True, label=f't={t[tp]:.1f}')

ax1.set_xlabel('Wealth')
ax1.set_ylabel('Density')
ax1.set_title('Household Wealth Distribution')
ax1.legend()

ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.set_title('Firm Value Distribution')
ax2.legend()

plt.tight_layout()
plt.show()

# Save the results of the simulation
np.savez('simulation_results.npz', solution=solution, t=t, gdp=gdp)



'''
# Parameters
params = [
  alpha:    Firm contribution to household income
  beta:     Household spending rate
  gamma:    Government spending rate
  zeta:     Central bank impact on household wealth
  mu:       Quadratic growth term for households
  delta:    Household contribution to firm revenue
  epsilon:  Firm expenses rate
  lambda:   Firm growth rate
  kay:      Carrying capacity for firms
  tau:      Tax rate
  pi:       Amplitude of cyclical policy changes
  phi:      Frequency of cyclical policy changes
  omega:    Central bank adjustment rate
  br:       Birth rate
  dr:       Death rate
  ir:       Immigration rate
  er:       Emigration rate
  nfr:      New firm rate
  brr:      Bankruptcy rate
  sigma_u:  Unemployment rate adjustment factor
  sigma_i:  Inflation rate adjustment factor
  sigma_r:  Interest rate adjustment factor
  sigma_t:  International trade balance adjustment factor
  rho:      Unemployment decay rate
  eta:      Impact of unemployment on household wealth
  theta:    Impact of stock market on household wealth
  kappa:    Impact of inflation on firm value
  xi:       Impact of international trade on firm value
  psi:      Impact of environmental factors on government spending
  chi:      Impact of interest rates on central bank policy
]

# State Variables
# y = [H, F, G, C, N_h, N_f, U, I, R, T, E, S]
# H: Total household wealth
# F: Total firm value
# G: Government spending
# C: Central bank assets
# N_h: Number of households
# N_f: Number of firms
# U: Unemployment rate
# I: Inflation rate
# R: Interest rate
# T: International trade balance
# E: Environmental impact
# S: Stock market index

# Agent Properties
# initial_value: Starting value for each agent (household or firm)
# growth_rate: Individual growth rate for each agent
# volatility: Volatility in the agent's value over time
'''