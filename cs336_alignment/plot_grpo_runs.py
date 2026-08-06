from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_FILES = {
    42: "results/standard_on_policy_r1_zero_20260805_214508/seed_42/metrics.jsonl",
    43: "results/standard_on_policy_r1_zero_20260805_214508/seed_43/metrics.jsonl",
    44: "results/standard_on_policy_r1_zero_20260806_121645/seed_44/metrics.jsonl",
    45: "results/standard_on_policy_r1_zero_20260806_121645/seed_45/metrics.jsonl",
}

STEP_COL = "step"

METRICS = {
    "train/loss": "Loss",
    "train/grad_norm": "Gradient norm",
    "train/token_entropy": "Token entropy",
    "train/reward": "Train total reward",
    "train/format_reward": "Train format reward",
    "val/reward": "Validation total reward",
    "val/format_reward": "Validation format reward",
    "val/avg_response_tokens": "Validation average response length",
}


def load_runs() -> pd.DataFrame:
    runs = []

    for seed, file_path in RUN_FILES.items():
        # Each line is a standalone JSON object, and evaluation rows contain
        # more keys than ordinary training rows.
        df = pd.read_json(file_path, lines=True)
        df["seed"] = seed
        runs.append(df)

    return pd.concat(runs, ignore_index=True)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_dir: str = "results/plots",
):
    metric_df = df[[STEP_COL, "seed", metric]].dropna()

    stats = (
        metric_df.groupby(STEP_COL)[metric]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values(STEP_COL)
    )

    # 作业给出的近似 95% 置信区间
    stats["ci"] = 1.96 * stats["std"] / np.sqrt(stats["count"])

    plt.figure(figsize=(8, 5))

    # 四个 seed 的实际曲线
    for seed, seed_df in metric_df.groupby("seed"):
        seed_df = seed_df.sort_values(STEP_COL)
        plt.plot(
            seed_df[STEP_COL],
            seed_df[metric],
            linewidth=1,
            alpha=0.35,
            label=f"seed {seed}",
        )

    # 四个 seed 的均值
    plt.plot(
        stats[STEP_COL],
        stats["mean"],
        linewidth=1.5,
        label="mean",
    )

    # 均值 ± 95% CI
    plt.fill_between(
        stats[STEP_COL],
        stats["mean"] - stats["ci"],
        stats["mean"] + stats["ci"],
        alpha=0.2,
        label="95% CI",
    )

    plt.xlabel("Rollout step")
    plt.ylabel(title)
    plt.title(f"{title} across 4 seeds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Metric names contain "/"; replace it so it is not treated as a directory.
    output_name = metric.replace("/", "_")
    plt.savefig(
        Path(output_dir) / f"{output_name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def main():
    df = load_runs()

    for metric, title in METRICS.items():
        if metric not in df.columns:
            print(f"Skip missing metric: {metric}")
            continue

        plot_metric(df, metric, title)


if __name__ == "__main__":
    main()
