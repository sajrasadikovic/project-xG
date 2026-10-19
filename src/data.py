import os
import pandas as pd
from understatapi import UnderstatClient

ID_LAUTARO = "7006"
SEASONS = [str(year) for year in range (2018, 2026)]

def download_player_shots(player_id):
    with UnderstatClient()as understat:
        shots = understat.player (player=str(player_id)).get_shot_data()
    df = pd.DataFrame(shots)
    df = df[df["season"].astype(str).isin(SEASONS)].copy()
    for col in ["X", "Y", "xG"]:
        df[col]=df[col].astype(float)
    df ["minute"]=df["minute"].astype(int)
    return df

if __name__ =="__main__":
    lautaro = download_player_shots(ID_LAUTARO)
    print ("Columns:", lautaro.columns.to_list())
    print(lautaro.head())
    print("Shots per season:")
    print(lautaro["season"].value_counts().sort_index())
    os.makedirs("data", exist_ok=True)
    lautaro.to_pickle("data/shots_lautaro.pkl")
    print("Saved in data/shots_lautaro.pkl")
