"""Grouped evaluation and channel-aware plots. No build_dataset dependency."""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dataset_config import (
    DATASET_PATH, FIGURES_DIR, METRICS_DIR, MATERIAL_ORDER,
    MATERIAL_NAMES, MATERIAL_COLORS, FEATURES, RANDOM_STATE,
)


def parse_boolean_column(series, nullable=False):
    mapping = {"true": True, "false": False, "1": True, "0": False,
               "1.0": True, "0.0": False}
    text = series.astype("string").str.strip().str.lower()
    known = text.isin(mapping)
    empty = text.isna() | text.isin(["", "none", "nan", "<na>"])
    if (~known & ~empty).any():
        raise ValueError(f"Invalid boolean values: {series[~known & ~empty].unique()}")
    result = text.map(mapping).astype("boolean")
    if not nullable and result.isna().any():
        raise ValueError(f"Missing boolean values in {series.name}")
    return result


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Run python src/signal_processing.py first")
    df = pd.read_csv(DATASET_PATH)
    required = set(FEATURES + ["energy", "file", "material", "experiment_id",
                               "channel", "analysis_role", "used_for_training", "irradiated"])
    if required - set(df.columns):
        raise ValueError(f"Rebuild dataset; missing columns: {sorted(required - set(df.columns))}")
    df["used_for_training"] = parse_boolean_column(df["used_for_training"])
    df["irradiated"] = parse_boolean_column(df["irradiated"], nullable=True)
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    valid = np.isfinite(df[FEATURES].to_numpy(dtype=float)).all(axis=1)
    if not valid.all():
        warnings.warn(f"Excluded {(~valid).sum()} rows with nonfinite features")
    df = df.loc[valid].reset_index(drop=True)
    if df.empty:
        raise ValueError("No finite feature rows")
    return df


def prepare_training_data(df):
    mask = (df["used_for_training"].fillna(False)
            & df["material"].isin(MATERIAL_ORDER)
            & ~df["irradiated"].fillna(False))
    train = df.loc[mask].copy().reset_index(drop=True)
    if set(train["material"]) != set(MATERIAL_ORDER):
        raise ValueError("Training must contain every material in MATERIAL_ORDER")
    counts = train.groupby("material")["experiment_id"].nunique()
    if counts.min() < 2:
        raise ValueError(f"Need >=2 independent series per material for grouped evaluation: {counts.to_dict()}")
    return train


def new_model(name, n_train):
    if name == "KNN":
        estimator = KNeighborsClassifier(n_neighbors=min(5, n_train), weights="distance")
    else:
        estimator = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                  max_iter=1000, early_stopping=False,
                                  random_state=RANDOM_STATE)
    return make_pipeline(StandardScaler(), estimator)


def evaluate_models(df_train):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = df_train[FEATURES], df_train["material"]
    groups = df_train["experiment_id"]
    n_splits = min(5, int(df_train.groupby("material")["experiment_id"].nunique().min()))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics, predictions, split_rows = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        train_groups, test_groups = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
        if train_groups & test_groups:
            raise RuntimeError("Group leakage")
        for indices, partition in ((train_idx, "train"), (test_idx, "test")):
            if set(y.iloc[indices]) != set(MATERIAL_ORDER):
                raise ValueError("A grouped fold lacks a class; revise experiment groups or split scheme")
            for group in sorted(set(groups.iloc[indices])):
                split_rows.append(dict(fold=fold, partition=partition, experiment_id=group))
        for name in ("KNN", "MLP"):
            model = new_model(name, len(train_idx))
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            metrics.append(dict(model=name, fold=fold,
                                accuracy=accuracy_score(y.iloc[test_idx], pred),
                                balanced_accuracy=balanced_accuracy_score(y.iloc[test_idx], pred)))
            rows = df_train.iloc[test_idx][["file", "experiment_id", "channel", "material"]].copy()
            rows["model"], rows["fold"], rows["predicted_material"] = name, fold, pred
            predictions.append(rows)
    metrics = pd.DataFrame(metrics)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(METRICS_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(METRICS_DIR / "evaluation_splits.csv", index=False)
    predictions.to_csv(METRICS_DIR / "out_of_fold_predictions.csv", index=False)
    for name in ("KNN", "MLP"):
        subset = predictions[predictions["model"] == name]
        report = classification_report(subset["material"], subset["predicted_material"],
                                       labels=MATERIAL_ORDER, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(METRICS_DIR / f"{name.lower()}_classification_report.csv")
        cm = confusion_matrix(subset["material"], subset["predicted_material"], labels=MATERIAL_ORDER)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=MATERIAL_ORDER).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}: grouped out-of-fold predictions")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{name.lower()}_confusion_matrix.png", dpi=220)
        plt.close(fig)
    print("\nGrouped evaluation:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return metrics, predictions


def fit_final_model(df_train):
    model = new_model("MLP", len(df_train))
    model.fit(df_train[FEATURES], df_train["material"])
    return model


def summarize_files(df):
    keys = ["file", "experiment_id", "material", "channel", "analysis_role"]
    return df.groupby(keys, as_index=False).agg(
        n_samples=("gamma", "size"), gamma_mean=("gamma", "mean"),
        gamma_std=("gamma", "std"), log_energy_mean=("log_energy", "mean"),
        log_energy_std=("log_energy", "std"))


def analyze_comparison_films(df, model):
    selection = ((df["analysis_role"] == "comparison")
                 | df["irradiated"].fillna(False))
    result = df.loc[selection & df["material"].isin(MATERIAL_ORDER)].copy()
    if result.empty:
        print("No comparison files found")
        return result, pd.DataFrame()
    result["predicted_material"] = model.predict(result[FEATURES])
    result["prediction_confidence"] = model.predict_proba(result[FEATURES]).max(axis=1)
    summary = summarize_files(result)
    majority = result.groupby("file")["predicted_material"].agg(lambda s: s.mode().iloc[0])
    summary["predicted_material"] = summary["file"].map(majority)
    result["matches_file_majority"] = result["predicted_material"].eq(result["file"].map(majority))
    result["matches_source_label"] = result["predicted_material"].eq(result["material"])
    for target, source in (("majority_fraction", "matches_file_majority"),
                           ("source_label_fraction", "matches_source_label"),
                           ("mean_prediction_confidence", "prediction_confidence")):
        summary[target] = summary["file"].map(result.groupby("file")[source].mean())
    summary["prediction_agreement"] = summary["material"].eq(summary["predicted_material"])
    result.to_csv(METRICS_DIR / "comparison_film_predictions.csv", index=False)
    summary.to_csv(METRICS_DIR / "comparison_file_summary.csv", index=False)
    verified = result.loc[result["irradiated"].fillna(False)]
    verified.to_csv(METRICS_DIR / "irradiated_film_predictions.csv", index=False)
    print("\nComparison files (one row per file; irradiation may be unknown):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return result, summary


def save_feature_plots(df):
    """Separate axes per material/channel; no categorical straight-line scatter."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    selected = df[df["material"].isin(MATERIAL_ORDER)].copy()
    if selected.empty:
        return
    materials = [m for m in MATERIAL_ORDER if m in set(selected["material"])]
    channels = sorted(selected["channel"].unique())
    summary = summarize_files(selected)
    summary.to_csv(METRICS_DIR / "file_feature_summary.csv", index=False)
    colors = {"baseline": "#4C72B0", "comparison": "#DD8452",
              "exploratory": "#8172B3", "reference": "#777777"}
    for feature, ylabel, suffix in (
        ("gamma", "Signed spectral slope gamma", "gamma"),
        ("log_energy", "log10(mean unnormalized FFT power)", "energy"),
    ):
        for style in ("mean_sd", "boxplot"):
            fig, axes = plt.subplots(len(materials), len(channels),
                                     figsize=(6 * len(channels), 3.5 * len(materials)),
                                     squeeze=False, sharey=False)
            for row, material in enumerate(materials):
                for col, channel in enumerate(channels):
                    ax = axes[row, col]
                    sub = selected[(selected["material"] == material)
                                   & (selected["channel"] == channel)]
                    order = sub[["file", "experiment_id", "analysis_role"]].drop_duplicates()
                    order = order.sort_values(["analysis_role", "experiment_id"])
                    if sub.empty:
                        ax.text(.5, .5, "No data", ha="center", transform=ax.transAxes)
                        continue
                    for position, record in enumerate(order.to_dict("records")):
                        color = colors.get(record["analysis_role"], "#777777")
                        values = sub.loc[sub["file"] == record["file"], feature].to_numpy()
                        if style == "mean_sd":
                            spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
                            ax.errorbar(position, values.mean(), yerr=spread,
                                        fmt="o", color=color, capsize=5, markersize=7)
                        else:
                            bp = ax.boxplot([values], positions=[position], widths=.55,
                                            patch_artist=True, showfliers=False, manage_ticks=False)
                            bp["boxes"][0].set_facecolor(color)
                    ax.set_xticks(range(len(order)))
                    ax.set_xticklabels(order["experiment_id"], rotation=35, ha="right")
                    ax.set_title(f"{MATERIAL_NAMES[material]} / {channel}")
                    ax.set_ylabel(ylabel)
                    ax.grid(axis="y", alpha=.25)
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            present = sorted(set(selected["analysis_role"]))
            handles = [Line2D([0], [0], marker="o", linestyle="", color=colors.get(role, "gray"), label=role)
                       for role in present]
            fig.legend(handles=handles, loc="upper center", ncol=len(handles))
            note = "Mean +/- within-file SD" if style == "mean_sd" else "Median / quartiles; outliers hidden"
            fig.suptitle(f"{note}; each panel uses its own y scale", y=.965)
            fig.tight_layout(rect=(0, 0, 1, .93))
            filename = (f"file_level_{suffix}_comparison.png" if style == "mean_sd"
                        else f"{suffix}_boxplots_by_file.png")
            fig.savefig(FIGURES_DIR / filename, dpi=220, bbox_inches="tight")
            plt.close(fig)


def analyze_clusters(df_train):
    scaled = StandardScaler().fit_transform(df_train[FEATURES])
    n_clusters = len(MATERIAL_ORDER)
    if len(np.unique(scaled, axis=0)) < n_clusters:
        return np.nan
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE).fit_predict(scaled)
    score = np.nan
    if 1 < len(np.unique(labels)) < len(labels):
        score = silhouette_score(scaled, labels, sample_size=min(5000, len(labels)), random_state=RANDOM_STATE)
    pd.DataFrame([dict(model="KMeans", silhouette_score=score)]).to_csv(
        METRICS_DIR / "clustering_metrics.csv", index=False)
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(scaled), min(20000, len(scaled)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(scaled[indices, 0], scaled[indices, 1], c=labels[indices], s=5, alpha=.3, cmap="viridis")
    ax.set(xlabel="Standardized gamma", ylabel="Standardized log_energy",
           title="Exploratory KMeans (clusters are not material labels)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kmeans_clusters.png", dpi=220)
    plt.close(fig)
    return score


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    df_train = prepare_training_data(df)
    print("Baseline series and channels (not a verified stability designation):")
    print(df_train.groupby(["material", "experiment_id", "channel"]).size().to_string())
    metrics, _ = evaluate_models(df_train)
    score = analyze_clusters(df_train)
    print(f"Exploratory silhouette: {score:.4f}")
    final_model = fit_final_model(df_train)
    analyze_comparison_films(df, final_model)
    save_feature_plots(df)
    print(f"Saved metrics: {METRICS_DIR}")
    print(f"Saved plots: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
