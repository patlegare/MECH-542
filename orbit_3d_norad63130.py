import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re

TLE_FILE = "sat000063130.txt"  

epochs, incl, raan, ecc, argp, mean_anom, mean_motion = [], [], [], [], [], [], []

with open(TLE_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Parse TLEs line-pair by line-pair
for i in range(0, len(lines), 2):
    try:
        line1 = lines[i].rstrip("\n")
        line2 = lines[i + 1].rstrip("\n")
        m = re.search(r"\s(\d{5}\.\d+)", line1)
        if not m:
            print(f"Epoch not found in: {line1}")
            continue
        epoch_field = m.group(1)             
        epoch_raw = float(epoch_field) # e.g., 25143.57154119
        integer_part = epoch_field.split('.')[0]  # e.g., "25143"
        year_two_digits = int(integer_part[:2])   # e.g., 25
        year = 2000 + year_two_digits  #good for 2025

        # Extract day-of-year + fractional day
        effective_epoch = epoch_raw - (year_two_digits * 1000)
        day_of_year = int(effective_epoch)
        frac_day = effective_epoch - day_of_year

        epoch_dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1 + frac_day)
        epochs.append(epoch_dt)

        # Orbital elements from line 2 
        incl.append(float(line2[8:16]))
        raan.append(float(line2[17:25]))
        ecc.append(float("0." + line2[26:33].strip()))
        argp.append(float(line2[34:42]))
        mean_anom.append(float(line2[43:51]))
        mean_motion.append(float(line2[52:63]))

    except Exception as e:
        print("Skipping malformed TLE pair:", e)
        continue

# Quick sanity print
print("\nFirst 5 parsed epochs:")
for t in epochs[:5]:
    print(t)

df = pd.DataFrame({
    "Epoch": epochs,
    "Inclination (deg)": incl,
    "RAAN (deg)": raan,
    "Eccentricity": ecc,
    "ArgPerigee (deg)": argp,
    "MeanAnomaly (deg)": mean_anom,
    "MeanMotion (rev/day)": mean_motion,
})

# Plot
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle(
    "Time Variation of Orbital Elements — Kosmos 2584 (NORAD 63130)",
    fontsize=30,        # Bigger title font
    fontweight="bold"
)

axs[0, 0].plot(df["Epoch"], df["Inclination (deg)"], label="Inclination")
axs[0, 0].set_ylabel("Inclination [°]")
axs[0, 0].legend()

axs[0, 1].plot(df["Epoch"], df["RAAN (deg)"], color='orange', label="RAAN")
axs[0, 1].set_ylabel("RAAN [°]")
axs[0, 1].legend()

axs[1, 0].plot(df["Epoch"], df["Eccentricity"], color='green', label="Eccentricity")
axs[1, 0].set_ylabel("Eccentricity")
axs[1, 0].legend()

axs[1, 1].plot(df["Epoch"], df["ArgPerigee (deg)"], color='red', label="Argument of Perigee")
axs[1, 1].set_ylabel("Arg. of Perigee [°]")
axs[1, 1].legend()

axs[2, 0].plot(df["Epoch"], df["MeanAnomaly (deg)"], color='purple', label="Mean Anomaly")
axs[2, 0].set_ylabel("Mean Anomaly [°]")
axs[2, 0].set_xlabel("Time")
axs[2, 0].legend()

axs[2, 1].plot(df["Epoch"], df["MeanMotion (rev/day)"], color='brown', label="Mean Motion")
axs[2, 1].set_ylabel("Mean Motion [rev/day]")
axs[2, 1].set_xlabel("Time")
axs[2, 1].legend()

plt.tight_layout()
plt.show()