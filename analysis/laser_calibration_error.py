"""
Laser Center Trace (No Alignment) — with Groups + Background Overlay

What it does
- Loads pre-cut and post-cut CSVs
- Pairs points by `idx` if present, else Hungarian/greedy
- Computes raw displacements (POST - PRE) with **no alignment**
- Writes overall and per-group stats and CSVs
- Plots: pairing lines and displacement vectors
- Optional background image (e.g., pre-scan) with adjustable alpha

Accepted CSV columns (any one coordinate system):
- Pixel:   ['cx_px','cy_px']
- World:   ['est_x','est_y']
- Generic: ['x','y']
Optional:  'idx' for stable pairing.

Outputs
- errors_table.csv (plus errors_table_<group>.csv)
- summary.txt (overall + each group)
- debug_matching.png (points + pairing lines)
- overlay_vectors.png (arrows from PRE → POST)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import warnings
import seaborn as sns
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Optional SciPy for Hungarian; falls back to greedy if unavailable
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    linear_sum_assignment = None
    HAS_SCIPY = False

# =============================
# ======== CONFIG =============
# =============================

PRE_CSV  = "/media/rp/Ubuntu_Data/rp_repo_ubn/hybrid_arm_mirror/data/laser_calib/0903/laser_centers_precut/centers.csv"
POST_CSV = "/media/rp/Ubuntu_Data/rp_repo_ubn/hybrid_arm_mirror/data/laser_calib/0903/laser_centers_postcut/centers.csv"

OUTDIR = "./analysis/laser_center_trace_output"
BACKGROUND_IMG = "/media/rp/Ubuntu_Data/rp_repo_ubn/hybrid_arm_mirror/data/laser_calib/0903/laser_centers_precut/01_enface_mean.png"  # e.g., "/path/to/pre_image.png" (pre-scan image)
BG_ALPHA = 0.35        # transparency of background overlay (0..1)

# Coordinate preference when multiple exist in CSV: 'est' (world) or 'px' (pixels)
PREFER = "est"
UNITS = "mm" if PREFER == "est" else "px"
INVERT_Y = (PREFER == "px")  # typical image coords

# Viz / stats
TOLERANCE = 0.25  # only used to draw circles; does not affect math
LABEL_ERRORS = False
CMAP = "viridis"
DPI = 300

# Grouping by ids
# Example: laser calibration test ids vs. OCT test ids
CALIB_ID_LIST = [1, 3, 7, 9]   # set [] to disable
OCT_ID_LIST   = []             # if empty, all non-calib -> 'oct'
# Color mode: 'group' to color by groups, 'magnitude' to color by |Δ|
COLOR_MODE = 'group'

# =============================
# ======== HELPERS ============
# =============================

def _validate_paths():
    missing = [p for p in [PRE_CSV, POST_CSV] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing input(s): {missing}")

def _load_csv(path, name):
    try:
        df = pd.read_csv(path)
        print(f"Loaded {name}: {len(df)} rows, {len(df.columns)} cols")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load {name} from {path}: {e}")


def _extract_xy(df, prefer="est"):
    # try preferred then fallbacks
    options = [("est", ["est_x","est_y"]), ("px", ["cx_px","cy_px"]), ("generic", ["x","y"])]
    if prefer != "est":
        options = [("px", ["cx_px","cy_px"]), ("est", ["est_x","est_y"]), ("generic", ["x","y"])]
    for tag, cols in options:
        if all(c in df.columns for c in cols):
            xy = df[cols].astype(float).to_numpy()
            ids = df["idx"].to_numpy() if "idx" in df.columns else None
            print(f"Using {tag} coordinates: {cols}")
            return xy, ids, tag
    raise ValueError(f"No valid x/y columns found. Available: {list(df.columns)}")


def _match_by_id(pre_ids, post_ids):
    lookup = {v: j for j, v in enumerate(post_ids)}
    pairs = []
    for i, v in enumerate(pre_ids):
        if v in lookup:
            pairs.append((i, lookup[v]))
    return pairs


def _match_hungarian(pre_xy, post_xy):
    if not HAS_SCIPY:
        return _match_greedy(pre_xy, post_xy)
    D = np.linalg.norm(pre_xy[:, None, :] - post_xy[None, :, :], axis=2)
    r, c = linear_sum_assignment(D)
    return list(zip(r, c))


def _match_greedy(pre_xy, post_xy):
    D = np.linalg.norm(pre_xy[:, None, :] - post_xy[None, :, :], axis=2)
    triples = [(D[i, j], i, j) for i in range(len(pre_xy)) for j in range(len(post_xy))]
    triples.sort()
    used_i, used_j, pairs = set(), set(), []
    for d, i, j in triples:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j)
        pairs.append((i, j))
        if len(pairs) >= min(len(pre_xy), len(post_xy)):
            break
    return pairs


def _analyze(errors):
    errors = np.asarray(errors)
    stats = {
        "count": int(len(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan"),
        "mean": float(np.mean(errors)) if len(errors) else float("nan"),
        "std": float(np.std(errors)) if len(errors) else float("nan"),
        "median": float(np.median(errors)) if len(errors) else float("nan"),
        "min": float(np.min(errors)) if len(errors) else float("nan"),
        "max": float(np.max(errors)) if len(errors) else float("nan"),
        "p95": float(np.percentile(errors, 95)) if len(errors) else float("nan"),
    }
    return stats

# =============================
# ======== PLOTTING ===========
# =============================

def _plot_pairings(pre_xy, post_xy, pairs, outdir):
    plt.figure(figsize=(8, 6))
    if BACKGROUND_IMG and os.path.exists(BACKGROUND_IMG):
        try:
            img = plt.imread(BACKGROUND_IMG)
            plt.imshow(img, alpha=BG_ALPHA)
        except Exception as e:
            print(f"[WARN] background load failed: {e}")
    plt.scatter(pre_xy[:,0], pre_xy[:,1], s=40, label="PRE", marker='o')
    plt.scatter(post_xy[:,0], post_xy[:,1], s=40, label="POST", marker='x')
    for i, j in pairs:
        plt.plot([pre_xy[i,0], post_xy[j,0]], [pre_xy[i,1], post_xy[j,1]], 'k-', alpha=0.3, lw=0.6)
    if INVERT_Y: plt.gca().invert_yaxis()
    plt.axis('equal'); plt.grid(True, alpha=0.3)
    plt.xlabel(f"X ({UNITS})"); plt.ylabel(f"Y ({UNITS})")
    plt.title("Pre vs Post (paired, no alignment)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "debug_matching.png"), dpi=DPI, bbox_inches='tight')
    plt.close()

def _plot_vectors(pre_xy_m, post_xy_m, ids, outdir, errors, vecs, group_labels=None):
    plt.figure(figsize=(10, 8))
    
    # Load background image and get its dimensions
    img_bounds = None
    if BACKGROUND_IMG and os.path.exists(BACKGROUND_IMG):
        try:
            img = plt.imread(BACKGROUND_IMG)
            # plt.imshow(img, alpha=BG_ALPHA)
            plt.imshow(img, alpha= BG_ALPHA)
            # Get image bounds with small border
            h, w = img.shape[:2]
            border = 5  # thin border in pixels
            img_bounds = {
                'x_min': -border,
                'x_max': w + border,
                'y_min': -border, 
                'y_max': h + border
            }
        except Exception as e:
            print(f"[WARN] background load failed: {e}")

    # Prepare coloring
    if COLOR_MODE == 'group' and group_labels is not None:
        palette = {'calib': 'tab:orange', 'oct': 'tab:blue', 'other': 'tab:gray'}
        colors = [palette.get(g, 'tab:gray') for g in group_labels]
        sm = None
    else:
        norm = plt.Normalize(vmin=float(np.min(errors)), vmax=float(np.max(errors)))
        cmap = cm.get_cmap(CMAP)
        colors = [cmap(norm(e)) for e in errors]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    head = 0.02 if UNITS == 'mm' else 2
    for k in range(len(errors)):
        c = colors[k]
        # Draw dotted black line connecting pre and post
        plt.plot([pre_xy_m[k,0], post_xy_m[k,0]], [pre_xy_m[k,1], post_xy_m[k,1]], 
                'k--', alpha=0.6, lw=1)
        plt.arrow(pre_xy_m[k,0], pre_xy_m[k,1], vecs[k,0], vecs[k,1],
                    head_width=head, length_includes_head=True, color=c, alpha=0.9, lw=1.8)
        if LABEL_ERRORS:
            mid = pre_xy_m[k] + 0.5*vecs[k]
            plt.text(mid[0], mid[1], f"{errors[k]:.3f}", fontsize=8,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
        # tolerance circle
        th = np.linspace(0, 2*np.pi, 100)
        cx = pre_xy_m[k,0] + TOLERANCE*np.cos(th)
        cy = pre_xy_m[k,1] + TOLERANCE*np.sin(th)
        # Tolerance circle is optional; uncomment to show
        # plt.plot(cx, cy, 'g--', alpha=0.45, lw=1)
        # id label
        plt.text(pre_xy_m[k,0] - 4, pre_xy_m[k,1] + 4, str(ids[k]), fontsize=15,
                ha='center', va='bottom', color='black')

    # points
    if COLOR_MODE == 'group' and group_labels is not None:
        for g in sorted(set(group_labels)):
            mask = np.array([gl == g for gl in group_labels])
            c = {'calib':'tab:orange','oct':'tab:blue','other':'tab:gray'}.get(g, 'tab:gray')
            if g == 'calib':
                plt.scatter(pre_xy_m[mask,0], pre_xy_m[mask,1], s=60, marker='o', edgecolors='black', linewidth=0.5, color='black', alpha=0)
                plt.scatter(post_xy_m[mask,0], post_xy_m[mask,1], s=60, marker='x', linewidth=2, color='#eb5e28', label='POST (Laser Calibration)')
            elif g == 'oct':
                plt.scatter(pre_xy_m[mask,0], pre_xy_m[mask,1], s=60, marker='o', edgecolors='black', linewidth=0.5, color='black', alpha=0)
                plt.scatter(post_xy_m[mask,0], post_xy_m[mask,1], s=60, marker='x', linewidth=2, color='blue', label='POST (OCT Alignment)')
    
    # Add single legend entry for all PRE points (black)
    if len(pre_xy_m) > 0:
        plt.scatter([], [], s=120, marker='o', edgecolors='black', linewidth=0.5, color='black', label='PRE cuts')
    else:
        plt.scatter(pre_xy_m[:,0], pre_xy_m[:,1], s=60, marker='o', label='PRE', edgecolors='black', linewidth=0.5)
        plt.scatter(post_xy_m[:,0], post_xy_m[:,1], s=60, marker='x', label='POST', linewidth=2)

    if COLOR_MODE != 'group' and sm is not None:
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.85); cbar.set_label(f'|Δ| ({UNITS})')

    if INVERT_Y: plt.gca().invert_yaxis()
    
    # Set axis limits to image bounds if available
    if img_bounds:
        plt.xlim(img_bounds['x_min'], img_bounds['x_max'])
        plt.ylim(img_bounds['y_min'], img_bounds['y_max'])
    else:
        plt.axis('equal')
    
    plt.axis('off')
    plt.legend(loc='lower right', fontsize=16); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overlay_vectors.png"), dpi=DPI, bbox_inches='tight')
    plt.close()

# =============================
# =========== MAIN ============
# =============================

def _group_labels_from_ids(matched_ids):
    calib_set = set(map(int, CALIB_ID_LIST)) if CALIB_ID_LIST else set()
    oct_set   = set(map(int, OCT_ID_LIST)) if OCT_ID_LIST else set()
    labels = []
    for v in map(int, matched_ids):
        if v in calib_set:
            labels.append('calib')
        elif (not OCT_ID_LIST and (not CALIB_ID_LIST or v not in calib_set)) or v in oct_set:
            # if OCT list empty: everything not calib → oct
            if OCT_ID_LIST:
                labels.append('oct' if v in oct_set else 'other')
            else:
                labels.append('oct')
        else:
            labels.append('other')
    return labels


def _write_group_csv_and_stats(df_out, errors, matched_ids, outdir):
    labels = _group_labels_from_ids(matched_ids)
    df_out['group'] = labels
    df_out.to_csv(os.path.join(outdir, 'errors_table.csv'), index=False)

    stats_all = _analyze(errors)
    groups = sorted(set(labels))
    # error column is last in df_out before 'group'
    err_col = [c for c in df_out.columns if c.startswith('error_')][0]
    group_stats = {g: _analyze(df_out.loc[df_out.group==g, err_col].to_numpy()) for g in groups}

    with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
        f.write("LASER CENTER TRACE SUMMARY (No Alignment)\n")
        f.write("="*44 + "\n\n")
        f.write(f"Points paired: {stats_all['count']}\n")
        f.write(f"Coordinate system: {PREFER} ({UNITS})\n\n")
        f.write("OVERALL:\n")
        f.write(f"  RMSE:   {stats_all['rmse']:.4f} {UNITS}\n")
        f.write(f"  Mean:   {stats_all['mean']:.4f} {UNITS}\n")
        f.write(f"  Median: {stats_all['median']:.4f} {UNITS}\n")
        f.write(f"  Std:    {stats_all['std']:.4f} {UNITS}\n")
        f.write(f"  Min:    {stats_all['min']:.4f} {UNITS}\n")
        f.write(f"  Max:    {stats_all['max']:.4f} {UNITS}\n")
        f.write(f"  P95:    {stats_all['p95']:.4f} {UNITS}\n\n")
        for g in groups:
            s = group_stats[g]
            f.write(f"GROUP '{g}' (n={s['count']}):\n")
            f.write(f"  RMSE:   {s['rmse']:.4f} {UNITS}\n")
            f.write(f"  Mean:   {s['mean']:.4f} {UNITS}\n")
            f.write(f"  Median: {s['median']:.4f} {UNITS}\n")
            f.write(f"  Std:    {s['std']:.4f} {UNITS}\n")
            f.write(f"  Min:    {s['min']:.4f} {UNITS}\n")
            f.write(f"  Max:    {s['max']:.4f} {UNITS}\n")
            f.write(f"  P95:    {s['p95']:.4f} {UNITS}\n\n")

    # also write split CSVs
    for g in groups:
        df_out[df_out.group==g].to_csv(os.path.join(outdir, f'errors_table_{g}.csv'), index=False)

    return labels


def _plot_error_boxplots(df_out: pd.DataFrame, outdir: str,
                            show_points: bool = True, ymax: float = 1.0):
    """
    Minimalist group-wise error plot.
    - Black box/whiskers
    - Colored points
    - No gridlines
    - Y-axis limited to [0, ymax]
    """
    import seaborn as sns

    # Find error column
    err_cols = [c for c in df_out.columns if c.startswith('error_')]
    if not err_cols:
        raise ValueError("No error_* column found in df_out.")
    err_col = err_cols[0]

    # Groups
    if 'group' not in df_out.columns:
        raise ValueError("'group' column missing. Did you call _write_group_csv_and_stats first?")
    order = [g for g in ['calib', 'oct', 'other'] if g in set(df_out['group'])]
    palette = {'calib': '#ff7f0e', 'oct': '#1f77b4', 'other': '#7f7f7f'}

    # Theme: minimal, no grid
    sns.set_theme(style="white", context="talk")

    plt.figure(figsize=(7, 5), dpi=DPI)

    # Black outline boxplots - made slimmer
    sns.boxplot(
        data=df_out, x="group", y=err_col, order=order,
        showcaps=True, boxprops=dict(facecolor='none', edgecolor='black'),
        whiskerprops=dict(color='black'), medianprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=4, linestyle='none', markeredgecolor='black'),
        width=0.5  # Make boxes slimmer
    )

    # Colored points on top
    if show_points:
        sns.stripplot(
            data=df_out, x="group", y=err_col, order=order,
            palette=palette, alpha=0.7, size=6, jitter=0.15
        )

    # Labels and limits
    plt.xlabel("")
    plt.ylabel("Calibration Error (mm)", fontsize=20)
    plt.ylim(0, ymax)
    # set y-ticks size

    # Annotate sample size under each group with custom labels
    ax = plt.gca()
    label_map = {'calib': 'Laser', 'oct': 'OCT', 'other': 'Other'}
    new_labels = []
    for g in order:
        n = (df_out['group'] == g).sum()
        display_name = label_map.get(g, g)
        new_labels.append(f"{display_name} (n={n})")
    ax.set_xticklabels(new_labels, fontsize=20)  # Reduced font size

    
    # Reduce y-axis font size
    ax.tick_params(axis='y', labelsize=20)

    sns.despine()
    plt.tight_layout()

    out_path = os.path.join(outdir, "errors_box_by_group.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"[OK] Saved minimalist box plot → {out_path}")




def main():
    print("Laser Center Trace (No Alignment)")
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    _validate_paths()

    pre_df  = _load_csv(PRE_CSV,  "PRE")
    post_df = _load_csv(POST_CSV, "POST")

    pre_xy,  pre_ids,  _ = _extract_xy(pre_df,  PREFER)
    post_xy, post_ids, _ = _extract_xy(post_df, PREFER)

    # Match
    if pre_ids is not None and post_ids is not None:
        pairs = _match_by_id(pre_ids, post_ids)
        if not pairs:
            raise RuntimeError("No overlapping idx IDs between PRE and POST")
        matched_ids = pre_ids[[i for i,_ in pairs]]
    else:
        pairs = _match_hungarian(pre_xy, post_xy)
        matched_ids = np.arange(len(pairs))

    pre_idx  = np.array([i for i,_ in pairs])
    post_idx = np.array([j for _,j in pairs])

    pre_m  = pre_xy[pre_idx]
    post_m = post_xy[post_idx]

    # Direct displacement (NO transform)
    vecs   = post_m - pre_m
    errors = np.linalg.norm(vecs, axis=1)

    # Save table with groups + write stats (overall + per group)
    df_out = pd.DataFrame({
        'id': matched_ids,
        f'pre_{PREFER}_x':  pre_m[:,0],
        f'pre_{PREFER}_y':  pre_m[:,1],
        f'post_{PREFER}_x': post_m[:,0],
        f'post_{PREFER}_y': post_m[:,1],
        'dx': vecs[:,0],
        'dy': vecs[:,1],
        f'error_{UNITS}': errors,
    }).sort_values('id')

    group_labels = _write_group_csv_and_stats(df_out, errors, matched_ids, OUTDIR)
    _plot_error_boxplots(df_out, OUTDIR, show_points=True, ymax=0.5)
    # (optional) violin version:
    # _plot_error_boxplots(df_out, OUTDIR, kind="violin", show_points=False)


    # Plots
    _plot_pairings(pre_m, post_m, pairs=[(i,i) for i in range(len(pre_m))], outdir=OUTDIR)
    _plot_vectors(pre_m, post_m, df_out['id'].to_numpy(), OUTDIR, errors, vecs, group_labels=group_labels)

    print("Done. Outputs in:", OUTDIR)

if __name__ == "__main__":
    main()
