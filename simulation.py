import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("predictions.csv", parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

time = df["Datetime"]
demand = df["y_pred"].values
hours = time.dt.hour.values

solar = np.zeros(len(df))

panel_efficiency = 0.70
cloud_variation = 0.3
peak_irradiance = 1.0

base_peak = 0.4 * demand.max()

for i, h in enumerate(hours):
    if 6 <= h <= 18:
        x = (h - 12) / 3.0
        irradiance = np.exp(-0.5 * x**2)
        cloud_factor = 1 - np.random.uniform(0, cloud_variation)
        solar[i] = base_peak * irradiance * panel_efficiency * cloud_factor

battery_capacity = 5.0
battery_max_power = 2.0
round_trip_eff = 0.90
charge_eff = np.sqrt(round_trip_eff)
discharge_eff = np.sqrt(round_trip_eff)

soc = battery_capacity / 2
soc_history = np.zeros(len(df))
battery_flow = np.zeros(len(df))
grid = np.zeros(len(df))


for i in range(len(df)):

    surplus = solar[i] - demand[i]

    if surplus > 0:
        charge = min(surplus,
                     battery_max_power,
                     battery_capacity - soc)

        soc += charge * charge_eff
        battery_flow[i] = -charge
        grid[i] = 0

    else:
        need = -surplus

        discharge = min(need,
                        battery_max_power,
                        soc)

        soc -= discharge
        battery_flow[i] = discharge * discharge_eff
        grid[i] = need - discharge * discharge_eff
    soc_history[i] = soc



from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


total_demand = demand.sum()
total_solar = solar.sum()
total_grid = grid.sum()
total_battery = np.abs(battery_flow).sum()

plt.figure()
plt.bar(["Demand", "Solar", "Grid"], 
        [total_demand, total_solar, total_grid])
plt.title("Energy Summary (kWh)")
plt.savefig(OUTPUT_DIR / "energy_summary.png", dpi = 300)
plt.show()

grid_price_day = 0.20
grid_price_night = 0.10

grid_cost = 0

for i, h in enumerate(hours):
    if 8 <= h <= 22:
        grid_cost += grid[i] * grid_price_day
    else:
        grid_cost += grid[i] * grid_price_night

no_solar_cost = np.sum(demand) * grid_price_day
savings = no_solar_cost - grid_cost

co2_per_kwh = 0.5
co2_without = np.sum(demand) * co2_per_kwh
co2_with = np.sum(grid) * co2_per_kwh
co2_saved = co2_without - co2_with

plt.figure()
plt.plot(time, grid)
plt.title("Grid Usage")
plt.savefig(OUTPUT_DIR / "grid usage.png", dpi = 300)
plt.show()

plt.figure()
plt.bar(["Without Solar", "With Solar"],
        [co2_without, co2_with])
plt.title("CO2 Emissions Comparison")
plt.savefig(OUTPUT_DIR / "with and without solar.png", dpi = 300)
plt.show()

print("Grid Cost With Solar: $", round(grid_cost,2))
print("Cost Without Solar: $", round(no_solar_cost,2))
print("Total Savings: $", round(savings,2))
print("CO2 Saved (kg):", round(co2_saved,2))

file_path = OUTPUT_DIR / "savings.txt"

with open(file_path, "w", encoding="utf-8") as file:
    file.write(f"Grid Cost With Solar: ${round(grid_cost, 2)}\n")
    file.write(f"Cost Without Solar: ${round(no_solar_cost, 2)}\n")
    file.write(f"Total Savings: ${round(savings, 2)}\n")
    file.write(f"CO2 Saved (kg): {round(co2_saved, 2)}\n")

print(f"File '{file_path}' created and written successfully.")