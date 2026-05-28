# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(
    "HESTIA_Simulation_Results.xlsx",
    sheet_name="Maastricht_2026-05-24"
)

x_peak = df["max_t_rect"]
x_pf   = df["t_rect_piek_postfinish"]
y      = df["cvr_co_reserve_min"]

# ----------------------------------
# Classificatie + grootte
# ----------------------------------
def classify_point(x, y):
    if (x > 40.5) and (y < 0):
        return "#8A00FF", 60     # FEL PAARS, GROOT
    elif (x > 40.5) or (y < 0):
        return "red", 15
    else:
        return "green", 15

colors_peak, sizes_peak = zip(*[classify_point(x, yy) for x, yy in zip(x_peak, y)])
colors_pf,   sizes_pf   = zip(*[classify_point(x, yy) for x, yy in zip(x_pf, y)])

# ----------------------------------
# Plot
# ----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Peak T_rect
axes[0].scatter(x_peak, y, c=colors_peak, s=sizes_peak, alpha=0.7)
axes[0].axvline(40.5, linestyle="--", linewidth=1)
axes[0].axhline(0, linestyle="--", linewidth=1)
axes[0].set_xlabel("T_rect max (°C)")
axes[0].set_ylabel("Cardiac Output Reserve (L/min)")
axes[0].set_title("Peak core temperature vs CO reserve")

# Post-finish
axes[1].scatter(x_pf, y, c=colors_pf, s=sizes_pf, alpha=0.7)
axes[1].axvline(40.5, linestyle="--", linewidth=1)
axes[1].axhline(0, linestyle="--", linewidth=1)
axes[1].set_xlabel("T_rect post-finish (°C)")
axes[1].set_title("Post-finish core temperature vs CO reserve")

plt.tight_layout()

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           label='Veilig',
           markerfacecolor='green', markersize=8),

    Line2D([0], [0], marker='o', color='w',
           label='Kritisch',
           markerfacecolor='red', markersize=8),

    Line2D([0], [0], marker='o', color='w',
           label='Potentieel EHS',
           markerfacecolor='#8A00FF', markersize=12)
]

axes[1].legend(
    handles=legend_elements,
    title="Risicocategorie",
    loc="upper right",
    frameon=True
)
plt.show()