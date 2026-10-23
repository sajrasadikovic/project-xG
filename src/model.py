import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import prepare_shots

LAUTARO_ID = "7006"
FEATURES = ["distance", "angle", "is_header", "is_open_play"]

def train_model(shots):
    X=shots[FEATURES]
    y = shots["is_goal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model=LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def load_lautaro_with_xg():
    """Train on the other players, tehn add my_xg to Lautaro's shots."""
    league_raw = pd.read_pickle("data/shots_league.pkl")
    league_shots, _ = prepare_shots(league_raw)
    others=league_shots[league_shots["player_id"].astype(str)!=LAUTARO_ID]
    model, _, _ = train_model(others)

    lautaro_raw=pd.read_pickle("data/shots_lautaro.pkl")
    lautaro, penalties =prepare_shots(lautaro_raw)
    lautaro["my_xg"]=model.predict_proba(lautaro[FEATURES])[:, 1]
    return lautaro, penalties

if __name__ =="__main__":
    league_raw=pd.read_pickle("data/shots_league.pkl")
    league_shots, _ =prepare_shots(league_raw)
    others=league_shots[league_shots["player_id"].astype(str) !=LAUTARO_ID]
    print("Training shots:", len(others))

    model, X_test, y_test = train_model(others)
    p_test=model.predict_proba(X_test)[:, 1]
    us_test=others.loc[y_test.index, "xG"].clip(1e-6, 1 - 1e-6)

    print("\n--- Test set (other players) ---")
    print("AUC         mine: ", round(roc_auc_score(y_test, p_test), 3),
         "| Understat:", round (roc_auc_score(y_test, us_test), 3))
    print("Log loss   mine:", round(log_loss(y_test, p_test), 3), 
          "| Understat:", round(log_loss(y_test, us_test), 3))
    print("Brier      mine:", round(brier_score_loss(y_test, p_test), 3),
          "| Understat:", round(brier_score_loss(y_test, us_test), 3))
    print ("Coefficients:", dict(zip(FEATURES, model.coef_[0].round(3))))

    lautaro_raw = pd.read_pickle("data/shots_lautaro.pkl")
    lautaro, _ =prepare_shots(lautaro_raw)
    lautaro["my_xg"]=model.predict_proba(lautaro[FEATURES])[:, 1]

    table=lautaro.groupby("season_label").agg(
        shots=("is_goal", "size"),
        goals=("is_goal", "sum"),
        my_xg=("my_xg", "sum"),
        understat_xg=("xG", "sum"),
    )

    table["goals_minus_my_xg"]=table["goals"]-table["my_xg"]
    print("\n--- Lautaro by season (no penalties) ---")
    print(table.round(2))
    print("\nTotal:")
    print(table.sum().round(2))
    print("\nCorrelation my xG vs Understat xG:", round(lautaro["my_xg"].corr(lautaro["xG"]), 3))

    frac, mean_pred = calibration_curve(y_test, p_test, n_bins=8, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac, marker="o", color="#0068A8", label="My model")
    plt.plot([0, 0.6], [0, 0.6], "--", color="grey", label="Perfect")
    plt.xlabel("Predicted xG")
    plt.ylabel("Actual goal rate")
    plt.legend()
    plt.legend()
    plt.savefig("assets/calibration.png", dpi=150, bbox_inches="tight")
    plt.show()