import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
BASELINE_DIR = Path("results/unitcost/MultiDismantler_real")
COMMUNITY_DIR = Path("results/community/MultiDismantler_real")
REPORT_PATH = Path("results/final_comparison_report.csv")
PLOTS_DIR = Path("results/plots")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def list_datasets(result_dir: Path):
    """Find dataset names from time&audc_*.csv recursively."""
    datasets = set()
    for csv_file in result_dir.rglob("time&audc_*.csv"):
        name = csv_file.stem.replace("time&audc_", "")
        datasets.add(name)
    return datasets


def find_csv(result_dir: Path, dataset: str):
    """Find time&audc file for a dataset."""
    candidates = list(result_dir.rglob(f"time&audc_{dataset}.csv"))
    return candidates[0] if candidates else None


def read_metrics(csv_path: Path, dataset: str):
    """
    Read time and audc from wide-format time&audc_*.csv.
    Header row contains dataset names; data rows contain time (row 0) and audc (row 1).
    We take the value under column == dataset.
    """
    try:
        df = pd.read_csv(csv_path, header=0)
        if dataset not in df.columns:
            return None, None
        time_val = df[dataset].iloc[0] if df.shape[0] >= 1 else None
        audc_val = df[dataset].iloc[-1] if df.shape[0] >= 1 else None
        return (float(time_val), float(audc_val))
    except Exception as e:
        print(f"[warn] failed to read {csv_path}: {e}")
        return None, None


def find_curve(result_dir: Path, dataset: str):
    """Find NormalizedLMCC file for a dataset (recursive)."""
    candidates = list(result_dir.rglob(f"NormalizedLMCC_{dataset}*"))
    return candidates[0] if candidates else None


def load_curve(curve_path: Path):
    try:
        with open(curve_path, "r") as f:
            return [float(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"[warn] failed to read curve {curve_path}: {e}")
        return None


def plot_curves(dataset: str, base_curve, comm_curve):
    max_len = max(len(base_curve), len(comm_curve))
    x_axis = np.linspace(0, 1, max_len)
    # pad to same length with last value
    if len(base_curve) < max_len:
        base_curve = base_curve + [base_curve[-1]] * (max_len - len(base_curve))
    if len(comm_curve) < max_len:
        comm_curve = comm_curve + [comm_curve[-1]] * (max_len - len(comm_curve))

    plt.figure()
    plt.plot(x_axis, base_curve, 'r-', label='Baseline')
    plt.plot(x_axis, comm_curve, 'b--', label='Community')
    plt.xlabel('Removed node ratio')
    plt.ylabel('Normalized LMCC')
    plt.title(f"Dismantling Curve: {dataset}")
    plt.legend()
    plt.tight_layout()
    out_path = PLOTS_DIR / f"{dataset}_curve.png"
    plt.savefig(out_path)
    plt.close()


def main():
    baseline_sets = list_datasets(BASELINE_DIR)
    community_sets = list_datasets(COMMUNITY_DIR)
    common = sorted(baseline_sets & community_sets)

    rows = []
    for ds in common:
        base_csv = find_csv(BASELINE_DIR, ds)
        comm_csv = find_csv(COMMUNITY_DIR, ds)
        if not base_csv or not comm_csv:
            print(f"[skip] missing csv for {ds}")
            continue
        base_time, base_audc = read_metrics(base_csv, ds)
        comm_time, comm_audc = read_metrics(comm_csv, ds)
        if base_audc is None or comm_audc is None or base_time is None or comm_time is None:
            print(f"[skip] invalid metrics for {ds}")
            continue
        diff = base_audc - comm_audc
        improv = (diff / base_audc * 100) if base_audc != 0 else 0.0
        time_diff = base_time - comm_time
        time_improv = (time_diff / base_time * 100) if base_time != 0 else 0.0
        rows.append({
            "Dataset": ds,
            "Baseline_AUDC": base_audc,
            "Community_AUDC": comm_audc,
            "Improvement_AUDC(%)": improv,
            "Baseline_Time": base_time,
            "Community_Time": comm_time,
            "Improvement_Time(%)": time_improv
        })

        # curves
        base_curve_file = find_curve(BASELINE_DIR, ds)
        comm_curve_file = find_curve(COMMUNITY_DIR, ds)
        if base_curve_file and comm_curve_file:
            base_curve = load_curve(base_curve_file)
            comm_curve = load_curve(comm_curve_file)
            if base_curve and comm_curve:
                plot_curves(ds, base_curve, comm_curve)
            else:
                print(f"[warn] invalid curves for {ds}")
        else:
            print(f"[warn] missing curve files for {ds}")

    if rows:
        df = pd.DataFrame(rows)
        df.sort_values(by="Improvement_AUDC(%)", ascending=False, inplace=True)
        print(df.to_string(index=False))
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(REPORT_PATH, index=False)
        print(f"\nReport saved to {REPORT_PATH}")
    else:
        print("No common datasets with valid audc found.")


if __name__ == "__main__":
    main()
