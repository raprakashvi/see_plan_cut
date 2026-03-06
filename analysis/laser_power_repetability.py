#!/usr/bin/env python3
"""
Folder Laser Power Analysis (formal labels, Channel A + Time stamp parsing, BW styling)

- Input layout:
    input_dir/
      Power_70/
        time_2_25.txt
      Power_100/
        time_2_00.txt
      ...

- For each txt (5 shots): parse (Time stamp, Channel A), detect shots, compute per-shot stats.
- Outputs:
    analysis/
      per_shot_summary.csv
      per_instance_summary.csv
      power_box_by_instance.png
      energy_box_by_instance.png
      combined_power_energy_boxplots.png
      per_file_plots/*power_vs_time.png
"""

import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Helpers: parsing + units
# ----------------------------
FLOAT_RE = r"^[\-\+]?(\d+(\.\d+)?|\.\d+)([eE][\-\+]?\d+)?$"

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def _guess_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Find 'Time stamp' and 'Channel A' (case/space/unit tolerant)."""
    candidates_time = []
    candidates_chan = []
    for c in df.columns:
        n = _norm(str(c))
        if n.startswith("timestamp") or n == "time" or n.startswith("time"):
            candidates_time.append(c)
        if "channel" in n and ("a" in n or n.endswith("channela")):
            candidates_chan.append(c)
        # also allow simple "a" or "ch a"
        if n in ("a", "cha", "channela"):
            candidates_chan.append(c)
    time_col = candidates_time[0] if candidates_time else None
    chan_col = candidates_chan[0] if candidates_chan else None
    return time_col, chan_col

def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert various time formats to seconds from start."""
    s = series.copy()
    # already numeric?
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.95:
        s = s_num
        return (s - s.iloc[0]).astype(float)

    # try datetime -> seconds since first
    dt = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
    if dt.notna().mean() > 0.8:
        secs = (dt - dt.iloc[0]).dt.total_seconds()
        return secs.astype(float)

    # try HH:MM:SS(.fff)
    td = pd.to_timedelta(s, errors="coerce")
    if td.notna().mean() > 0.8:
        secs = (td - td.iloc[0]).dt.total_seconds()
        return secs.astype(float)

    # last resort: coerce numeric
    s = pd.to_numeric(series, errors="coerce").fillna(method="ffill").fillna(0.0)
    return (s - s.iloc[0]).astype(float)

def _scale_units_from_name(colname: str) -> float:
    """Return multiplier to convert Channel A to Watts based on header."""
    n = _norm(colname)
    if "uw" in n or "µw" in colname.lower() or "μw" in colname.lower():
        return 1e-6
    if "mw" in n:
        return 1e-3
    # default assume Watts
    return 1.0

def load_time_power_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prefer parsing by headers: 'Time stamp' & 'Channel A'.
    Falls back to 'first two floats per line' if necessary.
    """
    # Try headered read with auto-sep
    try:
        df = pd.read_csv(
            path, sep=None, engine="python", comment=";",
            skip_blank_lines=True
        )
        # Drop all-empty cols/rows
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        # If we got a single unnamed column (bad split), try tab first
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep="\t", engine="python", comment=";", skip_blank_lines=True)
            df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

        time_col, chan_col = _guess_cols(df)
        if time_col is not None and chan_col is not None:
            t = _to_seconds(df[time_col])
            scale = _scale_units_from_name(chan_col)
            P = pd.to_numeric(df[chan_col], errors="coerce") * scale
            # clean
            mask = t.notna() & P.notna()
            t = t[mask].to_numpy(dtype=float)
            P = P[mask].to_numpy(dtype=float)
            # sort
            if t.size:
                idx = np.argsort(t)
                return t[idx], P[idx]
    except Exception:
        pass

    # Fallback: first two floats per line
    times, powers = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith((";", "!", "First Pulse")):
                continue
            parts = re.split(r"\s+", s.replace("\t", " "))
            floats = []
            for p in parts:
                if re.match(FLOAT_RE, p):
                    try:
                        floats.append(float(p))
                    except Exception:
                        pass
                if len(floats) == 2:
                    break
            if len(floats) == 2:
                times.append(floats[0])
                powers.append(floats[1])

    t = np.asarray(times, dtype=float)
    P = np.asarray(powers, dtype=float)
    if t.size:
        idx = np.argsort(t)
        t, P = t[idx], P[idx]
    return t, P

# ----------------------------
# Shot detection & stats
# ----------------------------
def detect_shots(
    t: np.ndarray,
    P: np.ndarray,
    thr_start: float = 0.5,
    thr_keep: float = 0.05,
    min_gap_s: float = 0.5,
    expect: int = 5,
) -> List[Tuple[int, int]]:
    """Two-threshold hysteresis; keep 'expect' longest segments, then by time."""
    if t.size == 0:
        return []
    above_start = P > thr_start
    starts = np.where(np.logical_and(above_start, np.concatenate([[False], ~above_start[:-1]])))[0]
    segs: List[Tuple[int, int]] = []
    for s_idx in starts:
        left = s_idx
        while left > 0 and P[left - 1] > thr_keep and (t[s_idx] - t[left - 1]) < 10.0:
            left -= 1
        right = s_idx
        while right < len(P) - 1 and P[right + 1] > thr_keep and (t[right + 1] - t[s_idx]) < 10.0:
            right += 1
        if segs and (t[left] - t[segs[-1][1]]) < min_gap_s:
            prev_l, prev_r = segs[-1]
            segs[-1] = (prev_l, max(prev_r, right))
        else:
            segs.append((left, right))
    segs = sorted(set(segs), key=lambda x: (x[0], x[1]))
    segs = sorted(segs, key=lambda lr: t[lr[1]] - t[lr[0]], reverse=True)[:expect]
    segs = sorted(segs, key=lambda lr: t[lr[0]])
    return segs

def integrate_energy(t_seg: np.ndarray, P_seg: np.ndarray) -> float:
    """Energy (J) = ∫ P dt ; negatives clamped to 0 before integration."""
    return float(np.trapz(np.maximum(P_seg, 0.0), t_seg))

def per_shot_stats(
    t: np.ndarray, P: np.ndarray, segments: List[Tuple[int, int]], thr_active: float
) -> pd.DataFrame:
    rows = []
    for i, (l, r) in enumerate(segments, start=1):
        t_seg = t[l : r + 1]
        P_seg = P[l : r + 1]
        mask = P_seg > thr_active
        P_act = P_seg[mask] if mask.any() else P_seg
        rows.append(dict(
            shot_id=i,
            t_start_s=float(t_seg[0]),
            t_end_s=float(t_seg[-1]),
            duration_s=float(t_seg[-1] - t_seg[0]),
            mean_power_W=float(P_act.mean()),
            std_power_W=float(P_act.std(ddof=1)) if len(P_act) > 1 else 0.0,
            energy_J=integrate_energy(t_seg, P_seg),
        ))
    return pd.DataFrame(rows)

# ----------------------------
# Styling (BW)
# ----------------------------
def set_bw_theme():
    sns.set(context="talk", style="white")
    plt.rcParams.update({
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "#BBBBBB",
        "text.color": "black",
    })

def unfill_boxes(ax):
    for patch in getattr(ax, "artists", []):
        patch.set_facecolor("none")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)
    for line in ax.lines:
        line.set_color("black")
        line.set_linewidth(1.2)

# ----------------------------
# Plotting
# ----------------------------
def plot_power_vs_time(t, P, segments, out_path: Path):
    set_bw_theme()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, P, linewidth=1.2, color="black")
    for i, (l, r) in enumerate(segments, start=1):
        ax.axvspan(t[l], t[r], alpha=0.15, color="#DDDDDD")
        ax.text((t[l] + t[r]) / 2, max(P[l:r + 1]) * 0.92, f"Shot {i}",
                ha="center", va="top", color="black", fontsize=10)
    ax.set_title("Power vs Time", color="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_box_by_instance(
    df: pd.DataFrame, x_col: str, y_col: str, out_path: Path,
    order: List[str], xtick_labels: List[str], y_label: str
):
    set_bw_theme()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, order=order,
                showcaps=True, whis=1.5, fliersize=3, color="white")
    sns.stripplot(data=df, x=x_col, y=y_col, ax=ax, order=order,
                  dodge=False, jitter=0.12, size=5, alpha=0.75, color="black")
    unfill_boxes(ax)
    ax.set_xlabel("Duty Cycle")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_combined_power_energy(df_power: pd.DataFrame, df_energy: pd.DataFrame,
                               instance_order: List[str], xtick_labels: List[str], out_path: Path):
    set_bw_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=False)

    # Power
    sns.boxplot(data=df_power, x="instance", y="mean_power_W", ax=axes[0], order=instance_order,
                showcaps=True, whis=1.5, fliersize=3, color="white")
    sns.stripplot(data=df_power, x="instance", y="mean_power_W", ax=axes[0], order=instance_order,
                  jitter=0.12, size=5, alpha=0.75, color="black")
    unfill_boxes(axes[0])
    axes[0].set_xlabel("Duty Cycle")
    axes[0].set_ylabel("Power (W)")
    axes[0].set_xticks(range(len(instance_order)))
    axes[0].set_xticklabels(xtick_labels, rotation=0)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Energy
    sns.boxplot(data=df_energy, x="instance", y="energy_J", ax=axes[1], order=instance_order,
                showcaps=True, whis=1.5, fliersize=3, color="white")
    sns.stripplot(data=df_energy, x="instance", y="energy_J", ax=axes[1], order=instance_order,
                  jitter=0.12, size=5, alpha=0.75, color="black")
    unfill_boxes(axes[1])
    axes[1].set_xlabel("Duty Cycle")
    axes[1].set_ylabel("Energy (J)")
    axes[1].set_xticks(range(len(instance_order)))
    axes[1].set_xticklabels(xtick_labels, rotation=0)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ----------------------------
# Label/ordering utilities
# ----------------------------
def parse_instance_label(power_dir: Path, file_path: Path) -> Tuple[str, str, float]:
    duty_match = re.search(r"Power[_\-]?(.+)$", power_dir.name, re.IGNORECASE)
    duty_str = duty_match.group(1) if duty_match else power_dir.name
    # filename: time_2_25.txt -> 2.25
    fname = file_path.stem
    t_match = re.match(r"time_(.+)$", fname)
    seconds = np.nan
    if t_match:
        t_part = t_match.group(1)
        if "_" in t_part:
            whole, frac = t_part.split("_", 1)
            frac = frac.replace("_", "")
            try:
                seconds = float(f"{whole}.{frac}")
            except Exception:
                pass
        else:
            try:
                seconds = float(t_part)
            except Exception:
                pass
    sec_str = f"{seconds:.2f}" if not np.isnan(seconds) else "?"
    instance_label = f"Power_{duty_str} | t={sec_str}s"
    return instance_label, duty_str, seconds

def extract_numeric(s: str) -> float:
    if s is None:
        return float("inf")
    m = re.search(r"[-+]?\d*\.?\d+", str(s))
    return float(m.group()) if m else float("inf")

# ----------------------------
# Main (config-style)
# ----------------------------
def main():
    # --- configure here ---
    input_dir = Path("/media/rp/Ubuntu_Data/OCT_Data/RATS/Laser_Power/Laser_Power_Test/9-7/")
    out_dir   = Path("./analysis/laser_power_repeatability/")
    thr_start = 0.5
    thr_keep  = 0.05
    min_gap_s = 0.5
    expected_shots = 5
    # ----------------------

    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = out_dir / "per_file_plots"
    per_file_dir.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.lower().startswith("power_")])
    if not subdirs:
        raise SystemExit(f"No subfolders like 'Power_*' found under: {input_dir}")

    per_shot_records = []
    per_instance_records = []

    for sd in subdirs:
        txts = sorted(sd.glob("time_*.txt"))
        if len(txts) != 1:
            print(f"[warn] Expected exactly one 'time_*.txt' in {sd}, found {len(txts)}; skipping.")
            continue
        txt = txts[0]

        instance_label, duty_str, seconds = parse_instance_label(sd, txt)
        duty_num = extract_numeric(duty_str)

        # Load by 'Time stamp' + 'Channel A' (fallback to generic)
        t, P = load_time_power_txt(txt)
        if t.size == 0:
            print(f"[warn] No (time, power) parsed from {txt}; skipping.")
            continue

        # Optional: sanity check max power (helps catch wrong column/units)
        # print(f"[debug] {sd.name}: max(P)={P.max():.3f} (W)")

        segments = detect_shots(t, P, thr_start=thr_start, thr_keep=thr_keep,
                                min_gap_s=min_gap_s, expect=expected_shots)
        if len(segments) != expected_shots:
            print(f"[info] {txt.name}: detected {len(segments)} shots (expected {expected_shots}).")

        df = per_shot_stats(t, P, segments, thr_active=thr_start)
        df["instance"] = instance_label
        df["duty_cycle"] = duty_str
        df["duty_num"] = duty_num
        df["time_s"] = seconds
        df["file"] = str(txt)
        per_shot_records.append(df)

        # Per-file line plot
        # plot_power_vs_time(t, P, segments, out_path=per_file_dir / f"{sd.name}__{txt.stem}__power_vs_time.png")

        # Instance summary across 5 shots
        per_instance_records.append({
            "instance": instance_label,
            "duty_cycle": duty_str,
            "duty_num": duty_num,
            "time_s": seconds,
            "mean_of_mean_power_W": df["mean_power_W"].mean(),
            "std_of_mean_power_W": df["mean_power_W"].std(ddof=1) if len(df) > 1 else 0.0,
            "mean_energy_J": df["energy_J"].mean(),
            "std_energy_J": df["energy_J"].std(ddof=1) if len(df) > 1 else 0.0,
            "n_shots": len(df),
        })

    if not per_shot_records:
        raise SystemExit("No data processed. Check your folder layout and filenames.")

    per_shot_df = pd.concat(per_shot_records, ignore_index=True)
    per_instance_df = pd.DataFrame(per_instance_records)

    # Save CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    per_shot_df.to_csv(out_dir / "per_shot_summary.csv", index=False)
    per_instance_df.to_csv(out_dir / "per_instance_summary.csv", index=False)

    # Ascending x-axis: duty_num -> time_s
    order_df = per_instance_df.copy()
    order_df["time_s_filled"] = order_df["time_s"].fillna(float("inf"))
    order_df = order_df.sort_values(by=["duty_num", "time_s_filled", "instance"], kind="mergesort")
    instance_order = order_df["instance"].tolist()
    xtick_labels = [str(int(extract_numeric(lbl.split(" | ")[0].split("_")[1]))) for lbl in instance_order]

    # Aggregated BW plots (no numbers on the figure)
    plot_box_by_instance(
        df=per_shot_df[["instance", "mean_power_W"]],
        x_col="instance", y_col="mean_power_W",
        out_path=out_dir / "power_box_by_instance.png",
        order=instance_order, xtick_labels=xtick_labels, y_label="Power (W)"
    )

    plot_box_by_instance(
        df=per_shot_df[["instance", "energy_J"]],
        x_col="instance", y_col="energy_J",
        out_path=out_dir / "energy_box_by_instance.png",
        order=instance_order, xtick_labels=xtick_labels, y_label="Energy (J)"
    )

    plot_combined_power_energy(
        df_power=per_shot_df[["instance", "mean_power_W"]],
        df_energy=per_shot_df[["instance", "energy_J"]],
        instance_order=instance_order, xtick_labels=xtick_labels,
        out_path=out_dir / "combined_power_energy_boxplots.png",
    )

    print("Saved outputs to:", out_dir)
    print("CSV files:")
    print(" -", out_dir / "per_shot_summary.csv")
    print(" -", out_dir / "per_instance_summary.csv")
    print("Figures:")
    print(" -", out_dir / "power_box_by_instance.png")
    print(" -", out_dir / "energy_box_by_instance.png")
    print(" -", out_dir / "combined_power_energy_boxplots.png")
    print(" -", out_dir / "per_file_plots/<...>power_vs_time.png")


if __name__ == "__main__":
    main()
