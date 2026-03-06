import pyvista as pv
import os
import numpy as np
import cv2
import scipy.signal as sig
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import scipy.optimize as opt
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl



def rmse(y_actual, y_pred):
    return np.sqrt(1/y_actual.shape[0] * np.sum((y_actual - y_pred) ** 2))


def fitFxn(xdata, mux, muy, sigma, A, d):
    '''
    A generic function that is used in parameter fitting. Replace with any function to be fitted.

    '''
    return -A * np.exp(-(0.5 * (((xdata[0] - mux) / sigma) ** 2 + ((xdata[1] - muy) / sigma) ** 2))) + d

def fitFxnSuper(xdata, mux, muy, sigma, A, d, P):
    '''
    A generic function that is used in parameter fitting. Replace with any function to be fitted.

    '''
    return -A * np.exp(-(0.5 * (((xdata[0] - mux) / sigma) ** 2 + ((xdata[1] - muy) / sigma) ** 2)) ** P) + d

def pv_pick(point):
        print(point)

def _ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def _make_grid(xmin, xmax, ymin, ymax, n=400):
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    return xg, yg

def _grid_from_points(points, pad=0.02, n=300):
    xmin, ymin = points[:,0].min(), points[:,1].min()
    xmax, ymax = points[:,0].max(), points[:,1].max()
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= pad * dx; xmax += pad * dx
    ymin -= pad * dy; ymax += pad * dy
    return _make_grid(xmin, xmax, ymin, ymax, n=n)

def _interp_to_image(xx, yy, vals, xg, yg):
    img = np.full_like(xg, np.nan, dtype=float)
    nx, ny = xg.shape[1], xg.shape[0]
    xmin, xmax = xg.min(), xg.max()
    ymin, ymax = yg.min(), yg.max()
    ix = np.clip(((xx - xmin) / max(xmax - xmin, 1e-9) * (nx - 1)).astype(int), 0, nx-1)
    iy = np.clip(((yy - ymin) / max(ymax - ymin, 1e-9) * (ny - 1)).astype(int), 0, ny-1)
    counts = {}
    sums = {}
    for i, j, v in zip(iy, ix, vals):
        key = (i, j)
        if key in sums:
            sums[key] += v
            counts[key] += 1
        else:
            sums[key] = v
            counts[key] = 1
    for (i, j), s in sums.items():
        img[i, j] = s / counts[(i, j)]
    return img

def _slice_band(points, axis="x", center=0.0, width=0.2):
    if axis == "x":
        mask = np.abs(points[:,0] - center) <= (width/2)
        xs = points[mask, 0]
        zs = points[mask, 2]
        ys = points[mask, 1]
        order = np.argsort(ys)
        return ys[order], zs[order]
    else:
        mask = np.abs(points[:,1] - center) <= (width/2)
        ys = points[mask, 1]
        zs = points[mask, 2]
        xs = points[mask, 0]
        order = np.argsort(xs)
        return xs[order], zs[order]





# Global figure style (safe for Overleaf)
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.transparent": True,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,  # embed as TrueType
    "ps.fonttype": 42,
})

def _equal_3d_axes(ax):
    # make 3D axes equal scale
    xlim = np.array(ax.get_xlim3d(), dtype=float)
    ylim = np.array(ax.get_ylim3d(), dtype=float)
    zlim = np.array(ax.get_zlim3d(), dtype=float)
    center = np.array([np.mean(xlim), np.mean(ylim), np.mean(zlim)])
    radius = max(np.diff(xlim), np.diff(ylim), np.diff(zlim))[0] / 2
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)

def _pretty_ticks0(ax, pts):
    # ticks starting from 0 relative to min
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    for axis, (mn, mx) in zip([ax.xaxis, ax.yaxis, ax.zaxis], zip(mins, maxs)):
        axis.set_major_locator(MaxNLocator(4))
    ax.set_xticklabels([f"{t - mins[0]:.2f}" for t in ax.get_xticks()])
    ax.set_yticklabels([f"{t - mins[1]:.2f}" for t in ax.get_yticks()])
    ax.set_zticklabels([f"{t - mins[2]:.2f}" for t in ax.get_zticks()])

def _param_box(ax, title, **params):
    txt = "\n".join([f"{k} = {v:.4g}" if isinstance(v,(int,float,np.floating)) else f"{k} = {v}"
                     for k,v in params.items()])
    at = AnchoredText(f"{title}\n{txt}", loc="upper right", prop=dict(size=9),
                      frameon=True, pad=0.3, borderpad=0.4)
    at.patch.set_alpha(0.8)
    ax.add_artist(at)

def _contour_levels(z, n=8):
    vmin, vmax = np.nanmin(z), np.nanmax(z)
    return np.linspace(vmin, vmax, n)

def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p_png = os.path.join(outdir, f"{name}.png")
    p_pdf = os.path.join(outdir, f"{name}.pdf")
    fig.tight_layout()
    fig.savefig(p_png, bbox_inches="tight", dpi=300)
    fig.savefig(p_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return p_png, p_pdf

def visualize_crater_fits_paper(points: np.ndarray,
                                colors: np.ndarray,
                                coefs: np.ndarray,
                                coefs_refined: np.ndarray,
                                coefs_sg_fixedP: np.ndarray = None,
                                P_fixed: float = None,
                                outdir: str = "./figs_paper",
                                mux: float = None,
                                muy: float = None,
                                slice_band_width: float = 0.5,
                                grid_N: int = 600,
                                view=(28, 48)):
    """
    Paper-ready visualization set. Uses your existing fit functions & parameters.
    Returns dict of all image paths.
    """
    _ensure_outdir(outdir)

    gx_mux, gx_muy, gx_sigma, gx_A, gx_d = [float(v) for v in coefs]
    if mux is None: mux = gx_mux
    if muy is None: muy = gx_muy
    sg_sigma, sg_A, sg_d, sg_P = [float(v) for v in coefs_refined]

    # Dense grid across the (possibly truncated) points
    xg, yg = _grid_from_points(points, pad=0.05, n=grid_N)

    # Surfaces: Gaussian & Super-Gaussian (free P)
    z_gauss = fitFxn([xg, yg], gx_mux, gx_muy, gx_sigma, gx_A, gx_d)
    z_super = fitFxnSuper([xg, yg], gx_mux, gx_muy, sg_sigma, sg_A, sg_d, sg_P)

    # Predictions at sample points (for residuals, RMSE)
    z_pred_gauss = fitFxn([points[:,0], points[:,1]], gx_mux, gx_muy, gx_sigma, gx_A, gx_d)
    z_pred_super = fitFxnSuper([points[:,0], points[:,1]], gx_mux, gx_muy, sg_sigma, sg_A, sg_d, sg_P)
    rmse_gauss = rmse(points[:,2], z_pred_gauss)
    rmse_super = rmse(points[:,2], z_pred_super)

    # Optional fixed-P model (if provided)
    have_fixedP = (coefs_sg_fixedP is not None) and (P_fixed is not None)
    if have_fixedP:
        z_pred_sgP = fitFxnSuper([points[:,0], points[:,1]],
                                 *coefs_sg_fixedP, P_fixed)
        rmse_sgP = rmse(points[:,2], z_pred_sgP)

    # ---------- 3D surfaces (Gaussian vs Super-Gaussian) ----------
    for name, Z, rmse_val, title in [
        ("surf3d_gaussian", z_gauss, rmse_gauss, "Gaussian Surface"),
        ("surf3d_super",    z_super, rmse_super, "Super-Gaussian Surface"),
    ]:
        fig = plt.figure(figsize=(4.8, 4.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xg, yg, -Z, linewidth=0, antialiased=True, alpha=0.7, color='#d62828')
        ax.scatter(points[:,0], points[:,1], -points[:,2], s=2, alpha=0.8, c=-points[:,2], cmap='viridis')
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(*view)
        _equal_3d_axes(ax)
        _pretty_ticks0(ax, np.column_stack([points[:,0], points[:,1], -points[:,2]]))
        _param_box(ax, f"{title}",
                   mux=gx_mux, muy=gx_muy,
                   sigma=(gx_sigma if "Gaussian" in title and "Super" not in title else sg_sigma),
                   A=(gx_A if "Gaussian Surface" in title else sg_A),
                   d=(gx_d if "Gaussian Surface" in title else sg_d),
                   P=(sg_P if "Super" in title else None),
                   RMSE=rmse_val)
        _save(fig, outdir, name)

    # ---------- 2D height map + contours + points ----------
    for name, Z, ttl in [
        ("height_gaussian", z_gauss, "Height Map — Gaussian"),
        ("height_super",    z_super, "Height Map — Super-Gaussian"),
    ]:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.pcolormesh(xg, yg, Z, shading="auto")
        cs = ax.contour(xg, yg, Z, levels=_contour_levels(Z), linewidths=0.6)
        ax.clabel(cs, inline=True, fmt="%.2f", fontsize=8)
        ax.scatter(points[:,0], points[:,1], s=2, c="k", alpha=0.15, label="samples")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(ttl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="z")
        ax.legend(loc="lower right", frameon=True)
        _save(fig, outdir, name)

    # ---------- Residual maps + Histogram inset ----------
    res_gauss = points[:,2] - z_pred_gauss
    res_super = points[:,2] - z_pred_super
    # Grid residual images (sparse → image)
    img_gauss = _interp_to_image(points[:,0], points[:,1], res_gauss, xg, yg)
    img_super = _interp_to_image(points[:,0], points[:,1], res_super, xg, yg)
    clim = max(np.nanmax(np.abs(img_gauss)), np.nanmax(np.abs(img_super)))
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    # for name, IMG, res_vec, ttl, rmse_val in [
    #     ("residual_gaussian", img_gauss, res_gauss, "Residuals — Gaussian", rmse_gauss),
    #     ("residual_super",    img_super, res_super, "Residuals — Super-Gaussian", rmse_super),
    # ]:
    #     fig, ax = plt.subplots(figsize=(4.6, 4.0))
    #     im = ax.imshow(IMG, origin="lower",
    #                    extent=[xg.min(), xg.max(), yg.min(), yg.max()],
    #                    aspect="equal", norm=norm, cmap="coolwarm")
    #     ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(ttl)
    #     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="z_obs − z_fit")
    #     # Histogram inset
    #     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    #     axins = inset_axes(ax, width="42%", height="42%", loc="upper right")
    #     axins.hist(res_vec, bins=40, density=True)
    #     axins.set_title(f"RMSE={rmse_val:.4f}", fontsize=8)
    #     axins.tick_params(labelsize=7)
    #     _save(fig, outdir, name)

    # ---------- Slice overlays at x≈μx and y≈μy ----------
    xs_gt, z_at_y_slice = _slice_band(points, axis="y", center=muy, width=slice_band_width)
    ys_gt, z_at_x_slice = _slice_band(points, axis="x", center=mux, width=slice_band_width)
    xs = np.linspace(points[:,0].min(), points[:,0].max(), 1200)
    ys = np.linspace(points[:,1].min(), points[:,1].max(), 1200)
    z_g_xslice = fitFxn([xs, np.full_like(xs, muy)], gx_mux, gx_muy, gx_sigma, gx_A, gx_d)
    z_sg_xslice = fitFxnSuper([xs, np.full_like(xs, muy)], gx_mux, gx_muy, sg_sigma, sg_A, sg_d, sg_P)
    z_g_yslice = fitFxn([np.full_like(ys, mux), ys], gx_mux, gx_muy, gx_sigma, gx_A, gx_d)
    z_sg_yslice = fitFxnSuper([np.full_like(ys, mux), ys], gx_mux, gx_muy, sg_sigma, sg_A, sg_d, sg_P)

    # X-slice
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(xs_gt, z_at_y_slice, linestyle='', marker='o', markersize=2, label="GT (band y≈μy)")
    ax.plot(xs, z_g_xslice, label=f"Gaussian (RMSE {rmse_gauss:.3f})")
    ax.plot(xs, z_sg_xslice, label=f"Super-Gaussian (RMSE {rmse_super:.3f})")
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_title("X-slice at y≈μy")
    ax.legend(frameon=True)
    _save(fig, outdir, "slice_x_overlay_paper")

    # Y-slice
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(ys_gt, z_at_x_slice, linestyle='', marker='o', markersize=2, label="GT (band x≈μx)",alpha=0.5, color='black')
    ax.plot(ys, z_g_yslice, label=f"Gaussian (RMSE {rmse_gauss:.3f})", alpha=0.7, color='#0077b6')
    ax.plot(ys, z_sg_yslice, label=f"Super-Gaussian (RMSE {rmse_super:.3f})",alpha=0.7, color='#d62828')
    ax.set_xlabel("y (mm)"); ax.set_ylabel("z (mm)"); 
    # ax.set_title("Y-slice at x≈μx")
    ax.legend(frameon=True,fontsize=8)
    _save(fig, outdir, "slice_y_overlay_paper")

    # ---------- Optional: fixed-P comparison panel ----------
    fixed_paths = {}
    if have_fixedP:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.hist(res_gauss, bins=60, alpha=0.6, label=f"Gaussian (RMSE {rmse_gauss:.3f})", density=True,color='#0077b6')
        ax.hist(res_super, bins=60, alpha=0.6, label=f"Super-Gaussian (RMSE {rmse_super:.3f})", density=True,color='#d62828')
        ax.hist(points[:,2] - z_pred_sgP, bins=60, alpha=0.6, label=f"Super-Gaussian (P={P_fixed:.2f}) (RMSE {rmse_sgP:.3f})", density=True,color='green')
        ax.set_xlabel("Residual (z_obs − z_fit)"); ax.set_ylabel("PDF")
        # ax.set_title("Residual Distributions")
        ax.legend(frameon=True, fontsize=8)
        fixed_paths["residual_hist_all"] = _save(fig, outdir, "residual_hist_all")[0]

    # ---------- Summary text (same as yours, tiny tweak for fixedP) ----------
    summary = {
        "gaussian": {
            "mux": gx_mux, "muy": gx_muy, "sigma": gx_sigma, "A": gx_A, "d": gx_d,
            "rmse": rmse_gauss
        },
        "super_gaussian": {
            "mux": gx_mux, "muy": gx_muy, "sigma": sg_sigma, "A": sg_A, "d": sg_d, "P": sg_P,
            "rmse": rmse_super
        }
    }
    if have_fixedP:
        summary["super_gaussian_fixedP"] = {
            "params": [float(v) for v in coefs_sg_fixedP], "P": float(P_fixed),
            "rmse": float(rmse_sgP)
        }
    with open(os.path.join(outdir, "fit_summary_paper.txt"), "w") as f:
        f.write(json.dumps(summary, indent=2))

    # Collect paths
    outputs = {
        "3d_gaussian": os.path.join(outdir, "surf3d_gaussian.png"),
        "3d_super": os.path.join(outdir, "surf3d_super.png"),
        "height_gaussian": os.path.join(outdir, "height_gaussian.png"),
        "height_super": os.path.join(outdir, "height_super.png"),
        "residual_gaussian": os.path.join(outdir, "residual_gaussian.png"),
        "residual_super": os.path.join(outdir, "residual_super.png"),
        "slice_x": os.path.join(outdir, "slice_x_overlay_paper.png"),
        "slice_y": os.path.join(outdir, "slice_y_overlay_paper.png"),
        "summary": os.path.join(outdir, "fit_summary_paper.txt"),
    }
    if have_fixedP:
        outputs.update(fixed_paths)
    return outputs



if __name__ == "__main__":
    #---------------------------
    path = r"/media/rp/Ubuntu_Data/OCT_Data/RATS/250911_OCT_Fitting/250911_LTI_fitting/250911_LTI_50/"
    index = 2 # Starting from 0
    P = 12.72661 # Fixed P value for super Gaussian fitting
    #--------------------------
    
    folders = os.listdir(path)
    # load data (points and colors both Nx3 array) 
    points = np.load(os.path.join(path, folders[index], "npy_xyz.npy"))
    colors = np.load(os.path.join(path, folders[index], "npy_rgb.npy"))
    
    # pl = pv.Plotter()
    # pl.add_mesh(points, scalars = colors)
    # pl.enable_point_picking(show_message='Right click to pick a point around the center of the crater and a second point just above the tissue plane...', callback = pv_pick)
    # pl.show()
    
    # xmean = float(input("X Mean (First Number in Array #1 Above): "))
    # ymean = float(input("Y Mean (Second Number in Array #1 Above): "))
    # cutHeight = float(input("Cut Height (Third Number in Array #2 Above): "))

    xmean = 1.918
    ymean = -3.528
    cutHeight = 0
    
    
    points = points[np.where(points[:,2] < cutHeight)[0]]
    colors = colors[np.where(points[:,2] < cutHeight)[0]]
    
    params = [xmean, ymean, 0.5, 2, np.median(points[:,2])] 
    coefs = opt.curve_fit(fitFxn, points[:,0:2].T, points[:,2].T, params, maxfev=10000)[0]

    coefs_sg_fixedP = opt.curve_fit(lambda x, *params_0: fitFxnSuper(x, *params_0, P), points[:,0:2].T, points[:,2].T, params, maxfev=10000)[0]
    # Fix negative sigma (sign doesn't matter since all appearances are squared)
    # ***FOR BIVARIATE ONLY!!!
    # if coefs[2] < 0:
    #     coefs[2] = -coefs[2]
    
    
    truncPoints = points[np.where(np.sqrt((points[:,0] - coefs[0]) ** 2 + (points[:,1] - coefs[1]) ** 2) < (coefs[2] * 2))[0]]
    truncColors = colors[np.where(np.sqrt((points[:,0] - coefs[0]) ** 2 + (points[:,1] - coefs[1]) ** 2) < (coefs[2] * 2))[0]]
    
    params = [coefs[2], coefs[3], coefs[4], 10]
    coefs_refined = opt.curve_fit(lambda x, *params_0: fitFxnSuper(x, coefs[0], coefs[1], *params_0), truncPoints[:,0:2].T, truncPoints[:,2].T, params, maxfev=10000)[0]
    
    xgrid, ygrid = np.mgrid[np.min(truncPoints[:,0]):np.max(truncPoints[:,0]):(np.max(truncPoints[:,0])-np.min(truncPoints[:,0]))/1000, np.min(truncPoints[:,1]):np.max(truncPoints[:,1]):(np.max(truncPoints[:,1])-np.min(truncPoints[:,1]))/1000]
    zgrid = fitFxnSuper([xgrid, ygrid], *coefs_sg_fixedP, P)
    zgrid_gauss = fitFxn([xgrid, ygrid], *coefs)
    rms = rmse(truncPoints[:,2], fitFxn(truncPoints[:,0:2].T, *coefs))
    rms_refined = rmse(truncPoints[:,2], fitFxnSuper(truncPoints[:,0:2].T, coefs[0], coefs[1], *coefs_refined))
    rms_sg = rmse(truncPoints[:,2], fitFxnSuper(truncPoints[:,0:2].T, *coefs_sg_fixedP, P))

    # pl = pv.Plotter()
    # pl.add_mesh(truncPoints, scalars = truncColors)
    # pl.add_mesh(pv.StructuredGrid(xgrid, ygrid, zgrid))
    # pl.show()

    pl = pv.Plotter()
    pl.add_mesh(truncPoints, scalars = truncColors)
    pl.add_mesh(pv.StructuredGrid(xgrid, ygrid, zgrid))
    pl.show()


    
    print("(Gaussian)--Scan #{}--\nA: {:.3f}, sigma: {:.3f}, rmse: {:.3f}".format(index + 1, coefs[3], coefs[2], rms))
    print("\tFor copy/pasting:", coefs[0], coefs[1], coefs[3], coefs[2], rms, sep = "\t")
    print("(Super-Gaussian)--Scan #{}--\nA: {:.3f}, sigma: {:.3f}, P: {:.3f}, rmse: {:.3f}".format(index + 1, coefs_refined[1], coefs_refined[0], coefs_refined[3], rms_refined))
    print("\tFor copy/pasting:", coefs_refined[1], coefs_refined[0], coefs_refined[3], rms_refined, sep = "\t")
    print("(Super-Gaussian with Fixed P)--Scan #{}----\nA: {:.3f}, sigma: {:.3f}, rmse: {:.3f}".format(index + 1, coefs_sg_fixedP[3], coefs_sg_fixedP[2], rms_sg))
    print("\tFor copy/pasting:", coefs_sg_fixedP[3], coefs_sg_fixedP[2], rms_sg, sep = "\t")
    
    # points: (N,3) array (x,y,z)
    # colors: (N,3) RGB in [0..255] or [0..1] (if you don’t have colors, pass np.zeros_like(points))
    # coefs: [mux, muy, sigma, A, d]           # Gaussian fit results
    # coefs_refined: [sigma, A, d, P]          # Super-Gaussian (center fixed to mux,muy)

  

    paper_paths = visualize_crater_fits_paper(
    points=truncPoints,
    colors=truncColors,
    coefs=coefs,
    coefs_refined=coefs_refined,
    coefs_sg_fixedP=coefs_sg_fixedP,   # optional; pass None if not needed
    P_fixed=P,                         # optional; pass None if not needed
    outdir="./figs_paper",
    mux=coefs[0],
    muy=coefs[1],
    slice_band_width=0.5,
    grid_N=600,
)
    print("Paper figures saved:", paper_paths)
