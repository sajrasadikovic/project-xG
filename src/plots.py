import os

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import VerticalPitch

from src.features import prepare_shots

def shot_map(df, title=""):
    """Draw the shots on the attacking half of the pitch. Circle size = xG, blue = goal, grey = no goal."""
    pitch = VerticalPitch(
        pitch_type="custom", pitch_length=105, pitch_width=68, half=True, pitch_color="#f4f1ea", line_color="#444444",
    )
    fig, ax = pitch.draw(figsize=(7, 6))

    goals = df[df["is_goal"]==1]
    misses = df[df["is_goal"]==0]

    pitch.scatter(misses["x_m"], misses["y_m"], s=misses["xG"]*900+40, c="#111111", edgecolors="white", alpha=0.45, ax=ax, label="No goal")
    pitch.scatter(goals["x_m"], goals["y_m"], s=goals["xG"]*900+40, c="#0A4D8C", edgecolors="white", ax=ax, label="Goal")

    ax.legend(loc="lower left")
    ax.set_title(title, fontsize=13)
    return fig 

if __name__=="__main__":
    raw = pd.read_pickle("data/shots_lautaro.pkl")
    shots, penalties = prepare_shots(raw)

    fig = shot_map(shots, "Lautaro Martinez - Serie A 2018/19 - 2025/26")
    os.makedirs("assets", exist_ok=True)
    fig.savefig("assets/shotmap.png", dpi=150, bbox_inches="tight")
    print ("Saved assets/shotmap.png")
    plt.show()