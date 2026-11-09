import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from src.features import prepare_shots
from src.model import FEATURES, LAUTARO_ID

BLUE, GREY = "#0068A8", "#9aa5b1"

def get_models():
    return{
        "Logistic regression (baseline)": LogisticRegression (max_iter=1000),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    }

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    league, _ = prepare_shots(pd.read_pickle("data/shots_league.pkl"))
    others=league[league["player_id"].astype(str)!=LAUTARO_ID]

    X, y = others[FEATURES], others["is_goal"]
    X_train, X_test, y_train, y_test= train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    print("Train:", len(X_train), "| Test:", len(X_test))

    us_test=others.loc[y_test.index, "xG"].clip(1e-6, 1 - 1e-6)

    rows, fitted = [], {}
    for name, model in get_models().items():
        print("Training", name, "...")
        cv=cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        p_train=model.predict_proba(X_train)[:, 1]
        p_test=model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "AUC train": roc_auc_score(y_train, p_train),
            "AUC test": roc_auc_score(y_test, p_test),
            "AUC CV mean": cv.mean(),
            "AUC std mean": cv.std(),
            "log loss train": log_loss(y_train, p_train),
            "log loss test": log_loss(y_test, p_test),
            "Brier test": brier_score_loss(y_test, p_test),
        } )
        fitted [name]=model

    rows.append({
        "model": "Understat xG (reference)",
        "AUC test": roc_auc_score(y_test, us_test),
        "log loss test": log_loss(y_test, us_test),
        "Brier test": brier_score_loss(y_test, us_test),
    })

    results = pd.DataFrame(rows).set_index("model").round(3)
    print(results.to_string())
    results.to_csv("results/model_comparison.csv")

    trained =results.drop(index="Understat xG (reference)")
    best_name=trained["log loss test"].idxmin()
    print("\nBest model (lowest test log loss):", best_name)

    x=np.arange(len(trained))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x-0.2, trained ["AUC train"], 0.4, label="Train", color=GREY)
    ax.bar(x+0.2, trained ["AUC test"], 0.4, label="Test", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(trained.index, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC (axis starts at 0.5)")
    ax.legend()
    fig.savefig("assets/model_train_vs_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, model in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=name)

    fpr, tpr, _ = roc_curve(y_test, us_test)
    ax.plot(fpr, tpr, "--", color="grey", label="Understat")
    ax.plot([0, 1], [0, 1], ":", color="black", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(fontsize=7)
    fig.savefig("assets/model_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    imp = permutation_importance(fitted[best_name], X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
    importance=pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    print("\nFeature importance:\n", importance.round(4).to_string())
    importance.round(4).to_csv("results/feature_importance.csv")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    importance.plot(kind="barh", ax=ax, color=BLUE)
    ax.set_xlabel("Drop in AUC when the feature is shuffled")
    ax.set_title(f"Feature importance: {best_name}")
    fig.savefig("assets/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    