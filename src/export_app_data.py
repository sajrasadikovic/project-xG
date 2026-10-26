import os 
import pandas as pd
from src.model import load_lautaro_with_xg

COLUMNS = ["season", "season_label", "minute", "x_m", "y_m", "distance", "angle", "xG", "my_xg", "is_goal", "result", "situation", "shotType"]

if __name__=="__main__":
    shots, penalties = load_lautaro_with_xg()
    penalties=penalties.copy()
    penalties["my_xg"]=float("nan")
    out=pd.concat([shots, penalties], ignore_index=True)[COLUMNS]

    os.makedirs("app_data", exist_ok=True)
    out.to_csv("app_data/lautaro_shots.csv", index=False)
    print(len(out), "rows saved in app_data/lautaro_shots.csv")