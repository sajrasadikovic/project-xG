import numpy as np
import pandas as pd

PITCH_LENGHT = 105
PITCH_WIDTH = 68
HALF_GOAL = 3.66

def add_features (df):
    df=df.copy()
    df["x_m"]=df["X"]*PITCH_LENGHT
    df["y_m"]=df["Y"]*PITCH_WIDTH

    dx = PITCH_LENGHT - df["x_m"]
    dy = PITCH_WIDTH / 2-df["y_m"]
    df["distance"]=np.hypot(dx,dy)

    df["angle"]=np.arctan2(2*HALF_GOAL*dx, dx**2+dy**2 -HALF_GOAL**2)
    df["is_goal"]=(df["result"]=="Goal").astype(int)
    df["is_header"]=(df["shotType"]=="Head").astype(int)
    df["is_open_play"]=(df["situation"]=="OpenPlay").astype(int)

    year= df["season"].astype(int)
    df["season_label"]=year.astype(str)+"/"+(year+1).astype(str).str[-2:]
    return df

def prepare_shots(df):
    df=df[df["result"]!="OwnGoal"]
    df=add_features(df)
    penalties = df[df["situation"]=="Penalty"]
    shots=df[df["situation"]!="Penalty"]
    return shots, penalties

if __name__=="__main__":
    raw = pd.read_pickle("data/shots_lautaro.pkl")
    shots, penalties = prepare_shots(raw)
    print(shots[["season_label", "distance", "angle", "xG", "is_goal"]].head())
    print(shots[["distance", "angle", "xG"]].describe().round(2))
    print(shots[["distance", "angle", "xG"]].corr().round(2))
    print("Penalties:", len (penalties), "- scored:", int(penalties["is_goal"].sum()))