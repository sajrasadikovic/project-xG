import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.features import prepare_shots
from src.model import LAUTARO_ID

BLUE, GREY = "#0A4D8C", "#111111"
OUT = "assets/eda"

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    league, _ = prepare_shots(pd.read_pickle("data/shots_league.pkl"))
    others = league[league["player_id"].astype(str)!=LAUTARO_ID].copy()
    lautaro, _=prepare_shots(pd.read_pickle("data/shots_lautaro.pkl"))
    lautaro=lautaro.copy()
    print("Other players:", len(others), "shots | Lautaro:", len(lautaro), "shots")

    print("\nOther players:")
    print(others[["distance", "angle", "xG"]].describe().round(3))
    print("\nLautaro:")
    print(others[["distance", "angle", "xG"]].describe().round(3))
    print("\nGoal share others:", round(others["is_goal"].mean(), 3))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(others["distance"], bins=30, density=True, alpha=0.6, color=GREY, label="Other players")
    ax.hist(lautaro["distance"], bins=30, density=True, alpha=0.6, color=BLUE, label="Other players")
    ax.set_xlabel("Distance from goal centre (m)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of shot distance")
    ax.legend()
    save (fig, "01_distance_distribution")

    bins=[0, 6, 9, 12, 16, 20, 25, 60]
    for df in (others, lautaro):
        df["dist_bin"]=pd.cut(df["distance"], bins=bins)
    rate_o=others.groupby("dist_bin", observed=True)["is_goal"].mean()
    rate_l=lautaro.groupby("dist_bin", observed=True)["is_goal"].mean()
    n_l=lautaro.groupby("dist_bin", observed=True).size()
    print("\nLautaro shots per distance bin:\n", n_l.to_string())

    x=np.arange(len(rate_o))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x-0.2, rate_o.values, 0.4, color=GREY, label="Other players")
    ax.bar(x+0.2, rate_l.reindex(rate_o.index), 0.4, color=BLUE, label="Lautaro")
    ax.set_xticks(x)
    ax.set_xticklabels(rate_o.index.astype(str), rotation=30)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Share of shots that are goals")
    ax.legend()
    save(fig, "02_goal_rate_by_distance")

    cols=["distance", "angle", "xG", "is_header", "is_open_play", "is_goal"]
    corr=others[cols].corr()
    print("\nCorrelations (other players): \n", corr.round(2))
    fig, ax=plt.subplots(figsize=(6, 5))
    im=ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation matrix")
    save(fig, "03_correlation")

    print("\nSaved graphs in", OUT)

