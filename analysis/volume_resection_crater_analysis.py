from __future__ import annotations
import numpy as np
import scipy.optimize as opt
import pyvista as pv

import sys
sys.path.append("./")  # to import from pa
from utils import downsampleOCT
from planner.cut_simulator import Simulator
import scipy.interpolate as interp

import os, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
import pyvista as pv



# ---------------- Theme / Aesthetics ----------------
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 450,
    "font.size": 9,
    "axes.spines.right": False, "axes.spines.top": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
COL_OBJ  = "#495057"   # target
COL_SIM  = "#f4a261"   # sim
COL_REAL = "#2a9d8f"   # real

OUTDIR = Path("./analysis/planner_figures"); OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utilities ----------------
def grid_regular(xyz, nx, ny, bbox=None, method="nearest"):
    """Grid scattered (x,y,z) → regular raster (nx,ny) with guaranteed fill."""
    xyz = np.asarray(xyz)[:, :3]
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    if bbox is None:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
    else:
        xmin, xmax, ymin, ymax = bbox
    X = np.linspace(xmin, xmax, nx)
    Y = np.linspace(ymin, ymax, ny)
    Xg, Yg = np.meshgrid(X, Y)
    Zg = griddata((x, y), z, (Xg, Yg), method=method)
    if np.any(~np.isfinite(Zg)):
        Znear = griddata((x, y), z, (Xg, Yg), method="nearest")
        Zg = np.where(np.isfinite(Zg), Zg, Znear)
    return Xg, Yg, Zg

def common_mask(*arrs):
    m = np.zeros_like(arrs[0], dtype=bool)
    for a in arrs: m |= ~np.isfinite(a)
    return ~m

# Volumes and directional errors (sign convention: more negative z = deeper cut)
def volumes_mm3(Zobj, Zsim, Zreal, cell_area_mm2):
    M = common_mask(Zobj, Zsim, Zreal)
    O, S, R = Zobj.copy(), Zsim.copy(), Zreal.copy()
    O[~M]=np.nan; S[~M]=np.nan; R[~M]=np.nan
    obj = -np.nansum(O) * cell_area_mm2
    sim = -np.nansum(S) * cell_area_mm2
    rea = -np.nansum(R) * cell_area_mm2
    # over/under relative to target (per-pixel differences)
    dS = S - O; dR = R - O
    sim_over   = -np.nansum(np.minimum(0, dS)) * cell_area_mm2   # deeper than target
    sim_under  =  np.nansum(np.maximum(0, dS)) * cell_area_mm2   # shallower than target
    real_over  = -np.nansum(np.minimum(0, dR)) * cell_area_mm2
    real_under =  np.nansum(np.maximum(0, dR)) * cell_area_mm2
    return (obj, sim, rea, sim_over, sim_under, real_over, real_under)

# --------- Concise volume bars with annotations ---------
def export_bar_volumes_concise(stats_nofb, stats_fb=None):
    """
    Overlay Objective/Simulation/Experimental at TWO x-slots:
      [0] 'w/o feedback'  -> from stats_nofb
      [1] 'w/ feedback'   -> empty (or from stats_fb if provided later)

    stats_* = (obj, sim, rea, sim_over, sim_under, real_over, real_under)
    """
    import numpy as np
    obj0, sim0, rea0, sim_over0, sim_under0, real_over0, real_under0 = stats_nofb

    if stats_fb is not None:
        obj1, sim1, rea1, sim_over1, sim_under1, real_over1, real_under1 = stats_fb
    else:
        obj1 = sim1 = rea1 = sim_over1 = sim_under1 = real_over1 = real_under1 = 0.0

    labels = ["w/o feedback", "w/ feedback"]
    x = np.arange(2)
    # Values per slot
    obj_vals = [obj0, obj1]
    sim_vals = [sim0, sim1]
    rea_vals = [rea0, rea1]

    # Figure & y-limit
    plt.figure(figsize=(6.0, 5.2))
    ymax = max(obj_vals + sim_vals + rea_vals) * 1.25 if max(obj_vals + sim_vals + rea_vals) > 0 else 1.0
    plt.ylim(0, ymax)

    # Draw per-slot overlays: Objective (widest), Simulation, Experimental (narrowest)
    def draw_overlay(xpos, v_obj, v_sim, v_rea):
        plt.bar(xpos, v_obj, width=0.70, color=COL_OBJ,  alpha=0.35, edgecolor=COL_OBJ,  linewidth=1.0, label=None)
        plt.bar(xpos, v_sim, width=0.55, color=COL_SIM,  alpha=0.65, edgecolor=COL_SIM,  linewidth=1.0, label=None)
        plt.bar(xpos, v_rea, width=0.40, color=COL_REAL, alpha=0.90, edgecolor=COL_REAL, linewidth=1.0, label=None)

    draw_overlay(0, obj0, sim0, rea0)
    draw_overlay(1, obj1, sim1, rea1)  # will be empty zeros if stats_fb is None

    # One-time legend handles (so we don't duplicate)
    handle_o = plt.bar(-1, 0, width=0.70, color=COL_OBJ,  alpha=0.35, edgecolor=COL_OBJ,  linewidth=1.0, label="Objective")
    handle_s = plt.bar(-1, 0, width=0.55, color=COL_SIM,  alpha=0.65, edgecolor=COL_SIM,  linewidth=1.0, label="Simulation")
    handle_r = plt.bar(-1, 0, width=0.40, color=COL_REAL, alpha=0.90, edgecolor=COL_REAL, linewidth=1.0, label="Experimental")
    plt.legend(frameon=False, fontsize=8, handles=[handle_o, handle_s, handle_r])

    plt.xticks(x, labels)
    plt.ylabel("Resected volume [mm³]")

    # Over/Under annotations — only for non-empty slot(s)
    if sim0 > 0 or rea0 > 0:
        txt_sim0  = f"-Over: {sim_over0:.1f}\n+Under: {sim_under0:.1f}"
        txt_real0 = f"-Over: {real_over0:.1f}\n+Under: {real_under0:.1f}"
        # approximate positions based on heights
        if sim0 > 0:
            plt.text(0, sim0 * 1.01, txt_sim0, ha='center', va='bottom', fontsize=8)
        if rea0 > 0:
            plt.text(0, rea0 * 1.01, txt_real0, ha='center', va='bottom', fontsize=8)

    if stats_fb is not None and (sim1 > 0 or rea1 > 0):
        txt_sim1  = f"-Over: {sim_over1:.1f}\n+Under: {sim_under1:.1f}"
        txt_real1 = f"-Over: {real_over1:.1f}\n+Under: {real_under1:.1f}"
        if sim1 > 0:
            plt.text(1, sim1 * 1.01, txt_sim1, ha='center', va='bottom', fontsize=8)
        if rea1 > 0:
            plt.text(1, rea1 * 1.01, txt_real1, ha='center', va='bottom', fontsize=8)

    plt.tight_layout(pad=0.3)
    plt.savefig(OUTDIR / "volumes_mm3_concise_overlay_2slots.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()



# --------- Directional tolerance (choose ONE metric: Excess Volume beyond τ) ---------
# We keep only **excess volume beyond τ [mm³]**, split into over/under, Sim/Real.
# d = Z - Z_target (mm); masks: over (d < -τ), under (d > τ)

def directional_excess_volume(Zobj, Zsim, Zreal, cell_area_mm2, bands_mm=(0.1, 0.2, 0.5, 1.0)):
    M = common_mask(Zobj, Zsim, Zreal)
    O, S, R = Zobj[M], Zsim[M], Zreal[M]
    dS = S - O
    dR = R - O
    rows = []  # (tau, sim_over_excess_mm3, sim_under_excess_mm3, real_over_excess_mm3, real_under_excess_mm3)
    for tau in bands_mm:
        so_exc = float(np.sum((-dS[dS < -tau] - tau))) * cell_area_mm2
        su_exc = float(np.sum(( dS[dS >  tau] - tau))) * cell_area_mm2
        ro_exc = float(np.sum((-dR[dR < -tau] - tau))) * cell_area_mm2
        ru_exc = float(np.sum(( dR[dR >  tau] - tau))) * cell_area_mm2
        rows.append((tau, so_exc, su_exc, ro_exc, ru_exc))
    return rows

# Plot: Excess volume beyond τ [mm³] (Sim/Real × Over/Under) per τ

def export_directional_excess_volume(rows, outname="directional_excess_volume.pdf"):
    taus = [f"±{t:g}" for t,*_ in rows]
    so_exc = [r[1] for r in rows]; su_exc = [r[2] for r in rows]
    ro_exc = [r[3] for r in rows]; ru_exc = [r[4] for r in rows]
    x = np.arange(len(taus)); w = 0.2
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(x-1.5*w, so_exc, width=w, color=COL_SIM,  alpha=0.95, label='Sim Over')
    ax.bar(x-0.5*w, su_exc, width=w, color=COL_SIM,  alpha=0.55, label='Sim Under')
    ax.bar(x+0.5*w, ro_exc, width=w, color=COL_REAL, alpha=0.95, label='Real Over')
    ax.bar(x+1.5*w, ru_exc, width=w, color=COL_REAL, alpha=0.55, label='Real Under')
    ax.set_xticks(x); ax.set_xticklabels(taus)
    ax.set_xlabel('Directional tolerance τ [mm]')
    ax.set_ylabel('Excess volume [mm³]')
    # ax.set_title('Directional tolerance (excess volume only)')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUTDIR / outname, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# --------- RMSE (overall and beyond τ) ---------
# RMSE (mm) across common support. For tolerance τ, we clamp residuals within ±τ to 0 by subtracting τ from |d|:
#   r_τ = sign(d) * max(|d|-τ, 0)

def rmse_metrics(Zobj, Zsim, Zreal, bands_mm=(0.0, 0.1, 0.2, 0.5, 1.0)):
    M = common_mask(Zobj, Zsim, Zreal)
    O, S, R = Zobj[M], Zsim[M], Zreal[M]
    dS = S - O
    dR = R - O
    target_mean_depth = float(np.mean(np.maximum(0.0, -O))) + 1e-12
    out = []  # (tau, rmse_sim_mm, rmse_sim_pct, rmse_real_mm, rmse_real_pct)
    for tau in bands_mm:
        def clamp(d):
            a = np.abs(d) - tau
            a[a < 0] = 0
            return np.sign(d) * a
        rS = dS if tau == 0.0 else clamp(dS)
        rR = dR if tau == 0.0 else clamp(dR)
        rmseS = float(np.sqrt(np.mean(rS**2)))
        rmseR = float(np.sqrt(np.mean(rR**2)))
        out.append((tau, rmseS, 100.0*rmseS/target_mean_depth, rmseR, 100.0*rmseR/target_mean_depth))
    return out

# Plot RMSE vs τ for Sim and Real (grouped bars). Annotate % of target mean depth under each bar.

def export_rmse_bars(rows, outname="rmse_vs_tau.pdf"):
    taus = [f"±{t:g}" if t > 0 else '0' for t, *_ in rows]
    sim_rmse = [r[1] for r in rows]
    sim_pct = [r[2] for r in rows]
    real_rmse = [r[3] for r in rows]
    real_pct = [r[4] for r in rows]
    x = np.arange(len(taus))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    b1 = ax.bar(x - w / 2, sim_rmse, width=w, color=COL_SIM, alpha=0.9, label='Simulation')
    b2 = ax.bar(x + w / 2, real_rmse, width=w, color=COL_REAL, alpha=0.9, label='Real')
    ax.set_xticks(x)
    ax.set_xticklabels(taus)
    ax.set_ylabel('RMSE [mm]')
    ax.set_xlabel('Tolerance τ [mm]')
    # ax.set_title('RMSE beyond τ (clamped residuals)')
    ax.legend(frameon=False, fontsize=8)
    ymax = max(ax.get_ylim()[1], max(sim_rmse + real_rmse) * 1.2)
    ax.set_ylim(0, ymax)
    # for bars, pcts in [(b1, sim_pct), (b2, real_pct)]:
    #     for rect, pct in zip(bars, pcts):
    #         ax.text(
    #             rect.get_x() + rect.get_width() / 2,
    #             rect.get_height() + 0.02 * ymax,
    #             f"{pct:.1f}%",
    #             ha='center',
    #             va='bottom',
    #             fontsize=8
    #         )
    fig.tight_layout(pad=0.3)
    fig.savefig(OUTDIR / outname, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------------- Main panel ----------------

def run_clinical_panel(preScan_transformed, postScan_transformed, objFxn, state,
                       Xlength_mm, Ylength_mm, Ascan, Bscan,
                       tolerance_bands=(0.1, 0.2, 0.5, 1.0)):
    # 1) Shared grid (filled); NOTE: everything assumed in mm
    bbox = (objFxn[:,0].min(), objFxn[:,0].max(), objFxn[:,1].min(), objFxn[:,1].max())
    Xg, Yg, Zobj = grid_regular(objFxn, Ascan, Bscan, bbox=bbox, method="nearest")
    _,  _,  Zsim = grid_regular(state,  Ascan, Bscan, bbox=bbox, method="nearest")
    _,  _,  Zreal= grid_regular(postScan_transformed, Ascan, Bscan, bbox=bbox, method="nearest")

    # 2) Volumes (unit-correct): area in mm²; volume in mm³
    cell_area_mm2 = (Xlength_mm * Ylength_mm) / (Ascan * Bscan)
    stats = volumes_mm3(Zobj, Zsim, Zreal, cell_area_mm2)
    export_bar_volumes_concise(stats)

    # done. 
    # 3) Directional tolerance in **units** (mm² area & mm³ excess volume)
    tol_rows_units = directional_excess_volume(Zobj, Zsim, Zreal, cell_area_mm2, bands_mm=tuple(tolerance_bands))
    # export_directional_excess_volume(tol_rows_units, outname="directional_excess_volume.pdf")
    export_directional_excess_volume(tol_rows_units, outname="directional_excess_volume.png")

    # 5) Save JSON for captions/tables
    summary = {
        "volumes_mm3": {
            "objective": stats[0], "simulation": stats[1], "real": stats[2],
            "simulation_overcut": stats[3], "simulation_undercut": stats[4],
            "real_overcut": stats[5], "real_undercut": stats[6],
        },
        "directional_tolerance_units": {
            f"±{t:g}mm": {
                "simulation": {"over_excess_mm3": so_v, "under_excess_mm3": su_v},
                "real":       {"over_excess_mm3": ro_v, "under_excess_mm3": ru_v},
            }
            for (t, so_v, su_v, ro_v, ru_v) in tol_rows_units
        },
        "grid": {"Ascan": Ascan, "Bscan": Bscan, "Xlength_mm": Xlength_mm, "Ylength_mm": Ylength_mm}
    }
    # 5) RMSE (raw and beyond τ)
    rmse_rows = rmse_metrics(Zobj, Zsim, Zreal, bands_mm=tuple(tolerance_bands))
    # export_rmse_bars(rmse_rows, outname="rmse_vs_tau.pdf")
    export_rmse_bars(rmse_rows, outname="rmse_vs_tau.png")


    # 6) Save JSON for captions/tables
    summary = {
        "volumes_mm3": {
            "objective": stats[0], "simulation": stats[1], "real": stats[2],
            "simulation_overcut": stats[3], "simulation_undercut": stats[4],
            "real_overcut": stats[5], "real_undercut": stats[6],
        },
        "directional_excess_volume": {
            f"±{t:g}mm": {"simulation": {"over_excess_mm3": so_v, "under_excess_mm3": su_v},
                          "real":       {"over_excess_mm3": ro_v, "under_excess_mm3": ru_v}}
            for (t, so_v, su_v, ro_v, ru_v) in tol_rows_units
        },
        "rmse_beyond_tau": [
            {"tau_mm": t, "simulation_rmse_mm": s, "simulation_pct_of_target_mean_depth": sp,
             "real_rmse_mm": r, "real_pct_of_target_mean_depth": rp}
            for (t, s, sp, r, rp) in rmse_rows
        ],
        "grid": {"Ascan": Ascan, "Bscan": Bscan, "Xlength_mm": Xlength_mm, "Ylength_mm": Ylength_mm}
    }
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("=== CLINICAL PANEL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved → {OUTDIR.resolve()}")




# Import pre and post, set initial variables

def plane(x, A, B, C):
    return A + B*x[0] + C*x[1]


# ---------------- INPUT PARAMETERS TO CHANGE ----------------------- #
sim = Simulator(1, 1.939, 1/0.333885349, 0.483187452, 11.071, 12.72661)
preScan = np.load(r"/media/rp/Ubuntu_Data/OCT_Data/RATS/Vol_Resection/prescan/20250912-112712/npy_xyz.npy")
postScan = np.load(r"/media/rp/Ubuntu_Data/OCT_Data/RATS/Vol_Resection/20250912-124541_no_fb/20250912-124541/npy_xyz.npy")
inputSeq = np.load(r"/media/rp/Ubuntu_Data/OCT_Data/RATS/Vol_Resection/inputSeq.npy")
xCenter = 3.577
yCenter = -3.57
Ascan = 512
Bscan = 256
totalArea = 7.154 * 7.14 # Xlength * Ylength
# ---------------- INPUT PARAMETERS TO CHANGE ----------------------- #

# Downsample PCDs
preScan = downsampleOCT(preScan)
postScan = downsampleOCT(postScan)

# Get plane fit from pre (or fkValues)
coefs = opt.curve_fit(plane, preScan[:,0:2].T, preScan[:,2])[0]
normal = np.array([-coefs[1], -coefs[2], 1.0]) / np.linalg.norm([-coefs[1], -coefs[2], 1.0])

# Define X-Y-Z axis in tissue frame
xAxis = np.array([1, 0, 0]) - np.dot(np.array([1, 0, 0]), normal) * normal
xAxis = xAxis / np.linalg.norm(xAxis)
zAxis = normal
yAxis = np.cross(zAxis, xAxis)

# Initial visualization of point clouds
pl = pv.Plotter()
pl.add_mesh(preScan, color = "green")
pl.add_mesh(postScan, color = 'red')
pl.add_mesh(pv.Arrow(start = (xCenter, yCenter, plane([xCenter, yCenter], *coefs)), direction = zAxis, scale='auto'), color='blue')
pl.add_mesh(pv.Arrow(start = (xCenter, yCenter, plane([xCenter, yCenter], *coefs)), direction = yAxis, scale='auto'), color='green')
pl.add_mesh(pv.Arrow(start = (xCenter, yCenter, plane([xCenter, yCenter], *coefs)), direction = xAxis, scale='auto'), color='red')
pl.add_mesh(pv.Arrow(start = (0, 0, 0), direction = (0, 0, 1), scale='auto'), color='blue')
pl.add_mesh(pv.Arrow(start = (0, 0, 0), direction = (0, 1, 0), scale='auto'), color='green')
pl.add_mesh(pv.Arrow(start = (0, 0, 0), direction = (1, 0, 0), scale='auto'), color='red')
pl.show()

# Transform both PCDs from OCT frame to tissue frame
R_tissue2OCT = np.hstack((xAxis[:,None], yAxis[:,None], zAxis[:,None])).T
tvec_tissue2OCT = -np.array([xCenter, yCenter, plane([xCenter, yCenter], *coefs)]) 
preScan_transformed = (R_tissue2OCT @ preScan.T + R_tissue2OCT @ np.repeat(tvec_tissue2OCT[:,None], preScan.shape[0], 1)).T
postScan_transformed = (R_tissue2OCT @ postScan.T + R_tissue2OCT @ np.repeat(tvec_tissue2OCT[:,None], postScan.shape[0], 1)).T

# -------------------- FILLER CODE TO FIX MINOR PROBLEMS WITH POSTSCAN TEMPORARILY -------------------------
postScan_transformed[:,2] = np.minimum(0, postScan_transformed[:,2]) + 0.3
# ----------------------------------------------------------------------------------------------------------

# Define objective using transformed preScan
objFxn = preScan_transformed.copy()
objFxn[np.where(np.logical_and(np.abs(preScan_transformed[:,0]) < 3, np.abs(preScan_transformed[:,1]) < 3))[0], 2] -= 2

# Simulate cuts on pre with inputSeq
state = preScan_transformed.copy()
for i in inputSeq:
    state = sim.simulate_SG(state, i)
    
# Sample sim and real to be in same grid as objective
postScan_interp = interp.NearestNDInterpolator(postScan_transformed[:,0:2], postScan_transformed[:,2], rescale = True)
postScan_transformed[:,2] = postScan_interp(objFxn[:,0], objFxn[:,1])
state_interp = interp.NearestNDInterpolator(state[:,0:2], state[:,2], rescale = True)
state[:,2] = state_interp(objFxn[:,0], objFxn[:,1])
    
#%% Plot simulated + obj + post and metrics
pl = pv.Plotter()
pl.add_mesh(postScan_transformed, color = "red")
pl.add_mesh(objFxn, color = "green")
pl.add_mesh(state, color = "blue")
pl.add_legend_scale()
pl.show()

print("Approx Objective Volume: {:.3f} mm3".format(-np.sum(objFxn[:,2]) * (totalArea / Ascan / Bscan)))
print("Approx Sim Volume: {:.3f} mm3".format(-np.sum(state[:,2]) * (totalArea / Ascan / Bscan)))
print("Approx Sim Overcut: {:.3f} mm3".format(-np.sum(np.minimum(0, state[:,2] - objFxn[:,2])) * (totalArea / Ascan / Bscan)))
print("Approx Sim Undercut: {:.3f} mm3".format(np.sum(np.maximum(0, state[:,2] - objFxn[:,2])) * (totalArea / Ascan / Bscan)))
print("Approx Real Volume: {:.3f} mm3".format(-np.sum(postScan_transformed[:,2]) * (totalArea / Ascan / Bscan)))
print("Approx Real Overcut: {:.3f} mm3".format(-np.sum(np.minimum(0, postScan_transformed[:,2] - objFxn[:,2])) * (totalArea / Ascan / Bscan)))
print("Approx Real Undercut: {:.3f} mm3".format(np.sum(np.maximum(0, postScan_transformed[:,2] - objFxn[:,2])) * (totalArea / Ascan / Bscan)))

# Build axes tuple for 3D arrows
axes_tissue = (xAxis, yAxis, zAxis)

# Call analysis + export:
run_clinical_panel(
    preScan_transformed=preScan_transformed,
    postScan_transformed=postScan_transformed,
    objFxn=objFxn,
    state=state,
    Xlength_mm=7.154,
    Ylength_mm=7.14,
    Ascan=Ascan,
    Bscan=Bscan
)
