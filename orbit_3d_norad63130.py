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
            continue
        epoch_field = m.group(1)
        epoch_raw = float(epoch_field)
        year_two_digits = int(epoch_field[:2])
        year = 2000 + year_two_digits

        effective_epoch = epoch_raw - (year_two_digits * 1000)
        day_of_year = int(effective_epoch)
        frac_day = effective_epoch - day_of_year
        epoch_dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1 + frac_day)
        epochs.append(epoch_dt)

        # Line 2 fields (fixed-width TLE columns)
        incl.append(float(line2[8:16]))
        raan.append(float(line2[17:25]))
        ecc.append(float("0." + line2[26:33].strip()))
        argp.append(float(line2[34:42]))
        mean_anom.append(float(line2[43:51]))     # still parsed but NOT plotted
        mean_motion.append(float(line2[52:63]))

    except Exception as e:
        print("Skipping malformed TLE pair:", e)
        continue

print("\nFirst 5 parsed epochs:")
for t in epochs[:5]:
    print(t)

df = pd.DataFrame({
    "Epoch": epochs,
    "Inclination (deg)": incl,
    "RAAN (deg)": raan,
    "Eccentricity": ecc,
    "ArgPerigee (deg)": argp,
    "MeanMotion (rev/day)": mean_motion,
})

# ---------- PLOTS (no Mean Anomaly subplot) ----------
plt.style.use('seaborn-v0_8-darkgrid')

# Use GridSpec so Mean Motion spans the full bottom row
fig = plt.figure(figsize=(12, 10))
fig.suptitle(
    "Time Variation of Orbital Elements — Kosmos 2584 (NORAD 63130)",
    fontsize=30, fontweight="bold", y=1.03
)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

ax00 = fig.add_subplot(gs[0, 0])   # Inclination
ax01 = fig.add_subplot(gs[0, 1])   # RAAN
ax10 = fig.add_subplot(gs[1, 0])   # Eccentricity
ax11 = fig.add_subplot(gs[1, 1])   # Arg of Perigee
ax20 = fig.add_subplot(gs[2, :])   # Mean Motion spans both columns

ax00.plot(df["Epoch"], df["Inclination (deg)"], label="Inclination")
ax00.set_ylabel("Inclination [°]")
ax00.legend()

ax01.plot(df["Epoch"], df["RAAN (deg)"], color='orange', label="RAAN")
ax01.set_ylabel("RAAN [°]")
ax01.legend()

ax10.plot(df["Epoch"], df["Eccentricity"], color='green', label="Eccentricity")
ax10.set_ylabel("Eccentricity")
ax10.legend()

ax11.plot(df["Epoch"], df["ArgPerigee (deg)"], color='red', label="Argument of Perigee")
ax11.set_ylabel("Arg. of Perigee [°]")
ax11.legend()

ax20.plot(df["Epoch"], df["MeanMotion (rev/day)"], color='brown', label="Mean Motion")
ax20.set_ylabel("Mean Motion [rev/day]")
ax20.set_xlabel("Time")
ax20.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()