import os
import numpy as np, pandas as pd

N=5000
RNG=np.random.default_rng(42)

def uncertainty(df, xg_col):
    goals=df["is_goal"].to_numpy()
    p=df[xg_col].to_numpy()
    observed = goals.sum()-p.sum()

    idx=RNG.integers(0, len(df), size=(N, len(df)))
    boot=(goals[idx]-p[idx]).sum(axis=1)
    low, high=np.percentile(boot, [2.5, 97.5])

    sim=(RNG.random((N, len(df)))<p).sum(axis=1)
    p_value=min(1.0, 2*min((sim>=goals.sum()).mean(), (sim<=goals.sum()).mean()))
    return observed, low, high, p_value

if __name__=="__main__":
    df=pd.read_csv("app_data/lautaro_shots.csv")
    shots=df[df["situation"]!="Penalty"]

    rows = []
    groups = [("Total", shots)]+list(shots.groupby("season_label"))
    for label, part in groups:
        for name, col in [("My xG", "my_xg"), ("Understat xG", "xG")]:
            obs, low, high, p = uncertainty(part, col)
            rows.append({
                "period": label, "xG source": name, "shots": len (part),
                "goals": int(part["is_goal"].sum()),
                "xG": round(part[col].sum(), 1),
                "goals minus xG": round (obs, 1),
                "95% low": round(low, 1), "95% high": round(high, 1),
                "p-value": round(p, 2),
            })

            table=pd.DataFrame(rows)
            print(table.to_string(index=False))
            os.makedirs("results", exist_ok=True)
            table.to_csv("results/uncertainty.csv", index=False)