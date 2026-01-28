import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load predictions
# ----------------------------
df = pd.read_csv("predictions.csv", parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

time = df["Datetime"]
demand = df["y_pred"].values   # predicted load

hours = time.dt.hour.values

# ----------------------------
# Solar bell-curve model
# ----------------------------

solar = np.zeros(len(df))

# peak solar output (tune this)
peak_solar = 0.8 * demand.max()

for i, h in enumerate(hours):
    if 6 <= h <= 18:
        x = (h - 12) / 3.0     # noon peak
        solar[i] = peak_solar * np.exp(-0.5 * x**2)

# ----------------------------
# Battery simulation
# ----------------------------

battery_capacity = 5.0   # kWh
soc = battery_capacity / 2   # initial state of charge

battery_flow = np.zeros(len(df))
grid = np.zeros(len(df))

for i in range(len(df)):

    surplus = solar[i] - demand[i]

    # Excess solar -> charge battery
    if surplus > 0:
        charge = min(surplus, battery_capacity - soc)
        soc += charge
        battery_flow[i] = -charge
        grid[i] = 0

    # Deficit -> discharge battery then grid
    else:
        need = -surplus
        discharge = min(need, soc)
        soc -= discharge
        battery_flow[i] = discharge
        grid[i] = need - discharge


# ----------------------------
# PLOTS (3 separate figures)
# ----------------------------

plt.figure()
plt.plot(time, demand)
plt.title("Predicted Demand vs Time")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.show()

plt.figure()
plt.plot(time, solar)
plt.title("Solar Generation vs Time")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.show()

plt.figure()
plt.plot(time, grid)
plt.title("Grid Power Needed vs Time")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.show()
