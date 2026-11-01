"""README figures (light and dark variants) from the trained model and the hold-out set.

    python -m olist_reviews.report
"""
import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from olist_reviews.config import FEATURES, MODEL_PATH, REPORTS_DIR, TARGET
from olist_reviews.features import load_features, temporal_split

FIGURES = REPORTS_DIR / "figures"
THEMES = {
    "light": {"bg": "#ffffff", "text": "#0f172a", "muted": "#57606a", "grid": "#e5e7eb", "bar": "#2563eb", "hi": "#dc2626"},
    "dark": {"bg": "#0d1117", "text": "#e6edf3", "muted": "#8b93a7", "grid": "#21262d", "bar": "#60a5fa", "hi": "#f87171"},
}


def _axes(theme: dict, title: str, size=(8, 4.2)):
    fig, ax = plt.subplots(figsize=size, dpi=150)
    fig.patch.set_facecolor(theme["bg"]); ax.set_facecolor(theme["bg"])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=theme["text"], fontsize=12, fontweight="bold", pad=12)
    return fig, ax


def _save(fig, name: str, mode: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}-{mode}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def delay_chart(df: pd.DataFrame, theme: dict, mode: str) -> None:
    bins = [-np.inf, -15, -8, -1, 0, 3, 7, 14, np.inf]
    labels = ["≤-15", "-14…-8", "-7…-1", "on day", "+1…3", "+4…7", "+8…14", ">14"]
    rate = df.groupby(pd.cut(df["delay_days"], bins, labels=labels), observed=False)[TARGET].mean()
    avg = df[TARGET].mean()
    fig, ax = _axes(theme, f"Share of 1–3★ reviews by delivery delay in days (dashed: average {avg:.0%})")
    colors = [theme["hi"] if i >= 4 else theme["bar"] for i in range(len(labels))]
    ax.bar(rate.index.astype(str), rate.values, color=colors, width=0.62)
    for x, v in enumerate(rate.values):
        ax.text(x, v + 0.015, f"{v:.0%}", ha="center", color=theme["text"], fontsize=9)
    ax.axhline(avg, color=theme["muted"], linestyle="--", linewidth=1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, min(1, rate.max() + 0.12))
    _save(fig, "delay", mode)


def decile_chart(y: pd.Series, proba: np.ndarray, theme: dict, mode: str) -> None:
    decile = pd.qcut(pd.Series(proba).rank(method="first", ascending=False), 10, labels=range(1, 11))
    rate = pd.Series(np.asarray(y)).groupby(decile.values, observed=False).mean()
    fig, ax = _axes(theme, "Hold-out orders ranked by predicted risk: share of 1–3★ reviews per decile")
    ax.bar([str(i) for i in rate.index], rate.values, color=theme["bar"], width=0.62)
    for x, v in enumerate(rate.values):
        ax.text(x, v + 0.012, f"{v:.0%}", ha="center", color=theme["text"], fontsize=9)
    base = float(np.mean(y))
    ax.axhline(base, color=theme["muted"], linestyle="--", linewidth=1)
    ax.text(9.4, base + 0.015, f"base rate {base:.0%}", ha="right", color=theme["muted"], fontsize=9)
    ax.set_xlabel("risk decile (1 = riskiest 10%)", color=theme["muted"], fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _save(fig, "deciles", mode)


def importance_chart(pipeline, X: pd.DataFrame, theme: dict, mode: str, top: int = 12) -> None:
    Xt = pipeline.named_steps["prep"].transform(X)
    contrib = pipeline.named_steps["model"].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=Xt.columns).sort_values().tail(top)
    fig, ax = _axes(theme, "What the model relies on: mean |SHAP| on the hold-out set", size=(8, 4.8))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    ax.barh(imp.index, imp.values, color=theme["bar"], height=0.6)
    ax.tick_params(axis="y", colors=theme["text"])
    ax.set_xlabel("mean absolute contribution (log-odds)", color=theme["muted"], fontsize=9)
    _save(fig, "importance", mode)


def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    df = load_features()
    _, test = temporal_split(df)
    proba = artifact["pipeline"].predict_proba(test[FEATURES])[:, 1]
    for mode, theme in THEMES.items():
        delay_chart(df, theme, mode)
        decile_chart(test[TARGET], proba, theme, mode)
        importance_chart(artifact["pipeline"], test[FEATURES], theme, mode)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
