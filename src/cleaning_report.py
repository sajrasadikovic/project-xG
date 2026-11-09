import os
import pandas as pd

KEY=["id", "X", "Y", "xG", "minute", "result", "situation", "shotType"]

def report (name, raw):
    print(f"\n=== {name}: {raw.shape[0]} rows, {raw.shape[1]} columns ===")
    missing = raw.isna().sum()
    print("Missing values per column:")
    print(missing[missing>0].to_string() if missing.any() else "none")
    print(raw.dtypes.to_string())

    bad_range=~(raw["X"].between(0, 1)&raw["Y"].between(0, 1) & raw["xG"].between(0, 1)&raw["minute"].between(0, 130))
    own_goals=raw["result"]=="OwnGoal"
    penalties=raw["situation"]=="Penalty"

    rows = [
        ("Rows in raw data", len (raw)),
        ("Duplicate shots (same id)", int(raw["id"].duplicated().sum())),
        ("Rows with missing key values", int (raw[KEY].isna().any(axis=1).sum())),
        ("Rows with unrealistic values", int (bad_range.sum())),
        ("Own goals (removed)", int(own_goals.sum())),
        ("Penalties (analysed separately)", int((penalties & ~own_goals).sum())),
        ("Rows used for the model", int((~own_goals & ~penalties).sum())),
    ]
    table = pd.DataFrame(rows, columns=["step", "rows"])
    print(table.to_string(index=False))
    return table

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for name, path in [("Other players", "data/shots_league.pkl"), 
                       ("Lautaro", "data/shots_lautaro.pkl")]:
        table = report(name, pd.read_pickle(path))
        table.to_csv(f"results/cleaning_{name.split()[0].lower()}.csv", index=False)
                        