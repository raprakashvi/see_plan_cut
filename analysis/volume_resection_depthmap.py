"""
This script is used to analyze the depth map of a volume resection.
"""
import numpy as np
import scipy.optimize as opt
import pyvista as pv
import sys
sys.path.append("./")
from utils import downsampleOCT
from planner.cut_simulator import Simulator
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def plane(x, A, B, C):
    return A + B*x[0] + C*x[1]

# Import pre and post, set initial variables

def unified_depth_limits(*point_sets):
    """Compute shared vmin/vmax for depth colormaps across multiple point clouds."""
    zmins = [p[:,2].min() for p in point_sets]
    zmaxs = [p[:,2].max() for p in point_sets]
    return float(min(zmins)), float(max(zmaxs))

def symmetric_limits(*arrays):
    """Symmetric limits around 0 for error maps / histograms."""
    m = max(float(np.nanmax(np.abs(a))) for a in arrays)
    return -m, m

def pct(numer, denom):
    return float('nan') if denom == 0 else 100.0 * float(numer) / float(denom)



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

# Calculate metrics
obj_volume = -np.sum(objFxn[:,2]) * (totalArea / Ascan / Bscan)
sim_volume = -np.sum(state[:,2]) * (totalArea / Ascan / Bscan)
sim_overcut = -np.sum(np.minimum(0, state[:,2] - objFxn[:,2])) * (totalArea / Ascan / Bscan)
sim_undercut = np.sum(np.maximum(0, state[:,2] - objFxn[:,2])) * (totalArea / Ascan / Bscan)
real_volume = -np.sum(postScan_transformed[:,2]) * (totalArea / Ascan / Bscan)
real_overcut = -np.sum(np.minimum(0, postScan_transformed[:,2] - objFxn[:,2])) * (totalArea / Ascan / Bscan)
real_undercut = np.sum(np.maximum(0, postScan_transformed[:,2] - objFxn[:,2])) * (totalArea / Ascan / Bscan)

print("Approx Objective Volume: {:.3f} mm3".format(obj_volume))
print("Approx Sim Volume: {:.3f} mm3".format(sim_volume))
print("Approx Sim Overcut: {:.3f} mm3".format(sim_overcut))
print("Approx Sim Undercut: {:.3f} mm3".format(sim_undercut))
print("Approx Real Volume: {:.3f} mm3".format(real_volume))
print("Approx Real Overcut: {:.3f} mm3".format(real_overcut))
print("Approx Real Undercut: {:.3f} mm3".format(real_undercut))

#%% ========== ENHANCED ANALYSIS AND PLOTTING SECTION ==========

# Set publication-quality plotting parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'DejaVu Sans'
})
plt.rcParams.update({
    'axes.prop_cycle': plt.cycler('color', ['#2E8B57','#4169E1','#DC143C','#FF6347','#FFD700']),
    'axes.spines.top': False, 'axes.spines.right': False,
})


# Define consistent color scheme
colors = {
    'objective': '#2E8B57',      # Sea Green
    'simulation': '#4169E1',     # Royal Blue  
    'experimental': '#DC143C',   # Crimson
    'overcut': '#FF6347',        # Tomato
    'undercut': '#FFD700'        # Gold
}

# Create comprehensive figure panel
def create_figure_panel():
    """Create a comprehensive figure panel for the paper"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
    
    # ===== PANEL A: 3D Surface Comparison =====
    ax_3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    
    # Create meshgrid for surface plotting
    x_range = np.linspace(objFxn[:,0].min(), objFxn[:,0].max(), 50)
    y_range = np.linspace(objFxn[:,1].min(), objFxn[:,1].max(), 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Interpolate surfaces
    obj_interp = interp.griddata(objFxn[:,0:2], objFxn[:,2], (X, Y), method='cubic')
    sim_interp = interp.griddata(state[:,0:2], state[:,2], (X, Y), method='cubic')
    exp_interp = interp.griddata(postScan_transformed[:,0:2], postScan_transformed[:,2], (X, Y), method='cubic')
    
    # Plot surfaces with transparency
    surf1 = ax_3d.plot_surface(X, Y, obj_interp, alpha=0.7, color=colors['objective'], label='Objective')
    surf2 = ax_3d.plot_surface(X, Y, sim_interp, alpha=0.6, color=colors['simulation'], label='Simulation')
    surf3 = ax_3d.plot_surface(X, Y, exp_interp, alpha=0.5, color=colors['experimental'], label='Experimental')
    
    ax_3d.set_xlabel('X (mm)', fontweight='bold')
    ax_3d.set_ylabel('Y (mm)', fontweight='bold')
    ax_3d.set_zlabel('Depth (mm)', fontweight='bold')
    ax_3d.set_title('A) 3D Surface Comparison', fontweight='bold', pad=20)
    
    # Custom legend for 3D plot
    proxy_obj = plt.Rectangle((0, 0), 1, 1, fc=colors['objective'], alpha=0.7)
    proxy_sim = plt.Rectangle((0, 0), 1, 1, fc=colors['simulation'], alpha=0.6)
    proxy_exp = plt.Rectangle((0, 0), 1, 1, fc=colors['experimental'], alpha=0.5)
    ax_3d.legend([proxy_obj, proxy_sim, proxy_exp], ['Objective', 'Simulation', 'Experimental'], 
                loc='upper left', bbox_to_anchor=(0, 1))
    
    # ===== PANEL B: Depth Maps =====
    depth_maps = [obj_interp, sim_interp, exp_interp]
    titles = ['B1) Objective', 'B2) Simulation', 'B3) Experimental']
    
    for i, (depth_map, title) in enumerate(zip(depth_maps, titles)):
        ax = fig.add_subplot(gs[0, 2+i])
        
        im = ax.contourf(X, Y, depth_map, levels=20, cmap='RdYlBu_r', extend='both')
        ax.contour(X, Y, depth_map, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Depth (mm)')
    
    # ===== PANEL C: Error Analysis =====
    # C1: Simulation vs Objective Error
    ax_c1 = fig.add_subplot(gs[1, 2])
    sim_error = state[:,2] - objFxn[:,2]
    error_map_sim = interp.griddata(objFxn[:,0:2], sim_error, (X, Y), method='cubic')
    
    im_c1 = ax_c1.contourf(X, Y, error_map_sim, levels=20, cmap='RdBu_r', extend='both')
    ax_c1.contour(X, Y, error_map_sim, levels=[0], colors='black', linewidths=2)
    ax_c1.set_title('C1) Simulation Error', fontweight='bold')
    ax_c1.set_xlabel('X (mm)')
    ax_c1.set_ylabel('Y (mm)')
    ax_c1.set_aspect('equal')
    
    divider = make_axes_locatable(ax_c1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_c1, cax=cax, label='Error (mm)')
    
    # C2: Experimental vs Objective Error
    ax_c2 = fig.add_subplot(gs[1, 3])
    exp_error = postScan_transformed[:,2] - objFxn[:,2]
    error_map_exp = interp.griddata(objFxn[:,0:2], exp_error, (X, Y), method='cubic')
    
    im_c2 = ax_c2.contourf(X, Y, error_map_exp, levels=20, cmap='RdBu_r', extend='both')
    ax_c2.contour(X, Y, error_map_exp, levels=[0], colors='black', linewidths=2)
    ax_c2.set_title('C2) Experimental Error', fontweight='bold')
    ax_c2.set_xlabel('X (mm)')
    ax_c2.set_ylabel('Y (mm)')
    ax_c2.set_aspect('equal')
    
    divider = make_axes_locatable(ax_c2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_c2, cax=cax, label='Error (mm)')
    
    # C3: Simulation vs Experimental Comparison
    ax_c3 = fig.add_subplot(gs[1, 4])
    sim_vs_exp_error = state[:,2] - postScan_transformed[:,2]
    error_map_comp = interp.griddata(objFxn[:,0:2], sim_vs_exp_error, (X, Y), method='cubic')
    
    im_c3 = ax_c3.contourf(X, Y, error_map_comp, levels=20, cmap='RdBu_r', extend='both')
    ax_c3.contour(X, Y, error_map_comp, levels=[0], colors='black', linewidths=2)
    ax_c3.set_title('C3) Sim vs Exp', fontweight='bold')
    ax_c3.set_xlabel('X (mm)')
    ax_c3.set_ylabel('Y (mm)')
    ax_c3.set_aspect('equal')
    
    divider = make_axes_locatable(ax_c3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_c3, cax=cax, label='Difference (mm)')
    
    # ===== PANEL D: Quantitative Analysis =====
    ax_d1 = fig.add_subplot(gs[2, 0:2])
    
    # Volume comparison bar chart
    volumes = [obj_volume, sim_volume, real_volume]
    volume_labels = ['Objective', 'Simulation', 'Experimental']
    volume_colors = [colors['objective'], colors['simulation'], colors['experimental']]
    
    bars = ax_d1.bar(volume_labels, volumes, color=volume_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax_d1.set_ylabel('Volume (mm³)', fontweight='bold')
    ax_d1.set_title('D1) Volume Comparison', fontweight='bold')
    ax_d1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, volume in zip(bars, volumes):
        height = bar.get_height()
        ax_d1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                  f'{volume:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # ===== PANEL D2: Error Metrics =====
    ax_d2 = fig.add_subplot(gs[2, 2:4])
    
    # Create grouped bar chart for overcut/undercut
    x_pos = np.arange(2)
    width = 0.35
    
    sim_errors = [sim_overcut, sim_undercut]
    exp_errors = [real_overcut, real_undercut]
    
    bars1 = ax_d2.bar(x_pos - width/2, sim_errors, width, label='Simulation', 
                     color=colors['simulation'], alpha=0.8, edgecolor='black')
    bars2 = ax_d2.bar(x_pos + width/2, exp_errors, width, label='Experimental', 
                     color=colors['experimental'], alpha=0.8, edgecolor='black')
    
    ax_d2.set_ylabel('Volume (mm³)', fontweight='bold')
    ax_d2.set_title('D2) Over/Under-cut Analysis', fontweight='bold')
    ax_d2.set_xticks(x_pos)
    ax_d2.set_xticklabels(['Overcut', 'Undercut'])
    ax_d2.legend()
    ax_d2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_d2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                      f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # ===== PANEL D3: Accuracy Metrics =====
    ax_d3 = fig.add_subplot(gs[2, 4])
    
    # Calculate accuracy metrics
    sim_rmse = np.sqrt(np.mean((state[:,2] - objFxn[:,2])**2))
    exp_rmse = np.sqrt(np.mean((postScan_transformed[:,2] - objFxn[:,2])**2))
    sim_mae = np.mean(np.abs(state[:,2] - objFxn[:,2]))
    exp_mae = np.mean(np.abs(postScan_transformed[:,2] - objFxn[:,2]))
    
    metrics = ['RMSE', 'MAE']
    sim_metrics = [sim_rmse, sim_mae]
    exp_metrics = [exp_rmse, exp_mae]
    
    x_pos = np.arange(len(metrics))
    bars1 = ax_d3.bar(x_pos - width/2, sim_metrics, width, label='Simulation', 
                     color=colors['simulation'], alpha=0.8, edgecolor='black')
    bars2 = ax_d3.bar(x_pos + width/2, exp_metrics, width, label='Experimental', 
                     color=colors['experimental'], alpha=0.8, edgecolor='black')
    
    ax_d3.set_ylabel('Error (mm)', fontweight='bold')
    ax_d3.set_title('D3) Accuracy Metrics', fontweight='bold')
    ax_d3.set_xticks(x_pos)
    ax_d3.set_xticklabels(metrics)
    ax_d3.legend()
    ax_d3.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_d3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ===== PANEL E: Statistical Analysis =====
    ax_e1 = fig.add_subplot(gs[3, 0:2])
    
    # Error distribution histograms
    bins = 30
    ax_e1.hist(sim_error, bins=bins, alpha=0.6, label='Simulation Error', 
              color=colors['simulation'], density=True, edgecolor='black')
    ax_e1.hist(exp_error, bins=bins, alpha=0.6, label='Experimental Error', 
              color=colors['experimental'], density=True, edgecolor='black')
    
    ax_e1.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax_e1.set_xlabel('Error (mm)', fontweight='bold')
    ax_e1.set_ylabel('Density', fontweight='bold')
    ax_e1.set_title('E1) Error Distribution', fontweight='bold')
    ax_e1.legend()
    ax_e1.grid(True, alpha=0.3)
    
    # ===== PANEL E2: Correlation Analysis =====
    ax_e2 = fig.add_subplot(gs[3, 2])
    
    # Scatter plot of simulation vs experimental
    scatter = ax_e2.scatter(state[:,2], postScan_transformed[:,2], alpha=0.6, 
                           c=objFxn[:,2], cmap='viridis', s=20)
    
    # Add perfect correlation line
    min_val = min(state[:,2].min(), postScan_transformed[:,2].min())
    max_val = max(state[:,2].max(), postScan_transformed[:,2].max())
    ax_e2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(state[:,2], postScan_transformed[:,2])[0,1]
    ax_e2.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax_e2.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    ax_e2.set_xlabel('Simulation Depth (mm)', fontweight='bold')
    ax_e2.set_ylabel('Experimental Depth (mm)', fontweight='bold')
    ax_e2.set_title('E2) Sim vs Exp Correlation', fontweight='bold')
    ax_e2.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax_e2, label='Objective Depth (mm)')
    
    # ===== PANEL E3: Residuals Analysis =====
    ax_e3 = fig.add_subplot(gs[3, 3:5])
    
    # Q-Q plot for normality assessment
    from scipy import stats
    
    sim_residuals = sim_error
    exp_residuals = exp_error
    
    # Sort residuals
    sim_sorted = np.sort(sim_residuals)
    exp_sorted = np.sort(exp_residuals)
    
    # Generate theoretical quantiles
    n_sim = len(sim_residuals)
    n_exp = len(exp_residuals)
    sim_quantiles = stats.norm.ppf(np.arange(1, n_sim+1) / (n_sim+1))
    exp_quantiles = stats.norm.ppf(np.arange(1, n_exp+1) / (n_exp+1))
    
    ax_e3.scatter(sim_quantiles, sim_sorted, alpha=0.6, label='Simulation', 
                 color=colors['simulation'], s=15)
    ax_e3.scatter(exp_quantiles, exp_sorted, alpha=0.6, label='Experimental', 
                 color=colors['experimental'], s=15)
    
    # Add reference line
    combined_residuals = np.concatenate([sim_residuals, exp_residuals])
    min_q, max_q = combined_residuals.min(), combined_residuals.max()
    ax_e3.plot([min_q, max_q], [min_q, max_q], 'k--', alpha=0.7, linewidth=2)
    
    ax_e3.set_xlabel('Theoretical Quantiles', fontweight='bold')
    ax_e3.set_ylabel('Sample Quantiles', fontweight='bold')
    ax_e3.set_title('E3) Q-Q Plot (Normality Check)', fontweight='bold')
    ax_e3.legend()
    ax_e3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Create and save the comprehensive figure in output folder
fig = create_figure_panel()
output_folder = './analysis/planner_figures'
os.makedirs(output_folder, exist_ok=True)

fig.savefig(os.path.join(output_folder, 'volumetric_resection_analysis.png'), dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
fig.savefig(os.path.join(output_folder, 'volumetric_resection_analysis.pdf'), bbox_inches='tight',
           facecolor='white', edgecolor='none')

plt.show()

#%% ========== ADDITIONAL SPECIALIZED FIGURES ==========

def create_process_workflow_figure():
    """Create a workflow diagram showing the process steps"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Volumetric Resection Planning Workflow', fontsize=20, fontweight='bold')
    
    # Step 1: Pre-operative scan
    ax = axes[0, 0]
    scatter = ax.scatter(preScan_transformed[:,0], preScan_transformed[:,1], 
                        c=preScan_transformed[:,2], cmap='viridis', s=20, alpha=0.7)
    ax.set_title('1) Pre-operative\nOCT Scan', fontweight='bold')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(scatter, ax=ax, label='Depth (mm)', shrink=0.8)
    
    # Step 2: Objective definition
    ax = axes[0, 1]
    scatter = ax.scatter(objFxn[:,0], objFxn[:,1], 
                        c=objFxn[:,2], cmap='RdYlBu_r', s=20, alpha=0.7)
    ax.set_title('2) Resection\nObjective', fontweight='bold')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(scatter, ax=ax, label='Target Depth (mm)', shrink=0.8)
    
    # Step 3: Planning sequence
    ax = axes[0, 2]
    # Visualize input sequence as a heatmap
    seq_grid = np.zeros((10, 10))  # Assuming 10x10 grid for visualization
    
    # Handle inputSeq safely - check if it's scalar or array
    if np.isscalar(inputSeq) or len(inputSeq.shape) == 0:
        # If inputSeq is a scalar, create a simple pattern
        for i in range(min(100, seq_grid.size)):
            row, col = divmod(i, 10)
            if row < 10:
                seq_grid[row, col] = float(inputSeq) if np.isscalar(inputSeq) else float(inputSeq.item())
    else:
        # If inputSeq is an array, use its values
        for i, seq_val in enumerate(inputSeq.flat[:100]):  # Limit to first 100 for visualization
            row, col = divmod(i, 10)
            if row < 10:
                try:
                    seq_grid[row, col] = float(seq_val)
                except (ValueError, TypeError):
                    seq_grid[row, col] = 0.0  # Default value if conversion fails
    
    im = ax.imshow(seq_grid, cmap='plasma', aspect='equal')
    ax.set_title('3) Planned Cut\nSequence', fontweight='bold')
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    plt.colorbar(im, ax=ax, label='Cut Parameter', shrink=0.8)
    
    # Step 4: Simulation
    ax = axes[0, 3]
    scatter = ax.scatter(state[:,0], state[:,1], 
                        c=state[:,2], cmap='coolwarm', s=20, alpha=0.7)
    ax.set_title('4) Simulation\nResult', fontweight='bold')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(scatter, ax=ax, label='Simulated Depth (mm)', shrink=0.8)
    
    # Step 5: Experimental execution
    ax = axes[1, 0]
    scatter = ax.scatter(postScan_transformed[:,0], postScan_transformed[:,1], 
                        c=postScan_transformed[:,2], cmap='coolwarm', s=20, alpha=0.7)
    ax.set_title('5) Experimental\nExecution', fontweight='bold')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.colorbar(scatter, ax=ax, label='Actual Depth (mm)', shrink=0.8)
    
    # Step 6: Validation metrics
    ax = axes[1, 1]
    exp_rmse = np.sqrt(np.mean((postScan_transformed[:,2] - objFxn[:,2])**2))
    metrics_data = [
        ['Volume Accuracy', f'{(1 - abs(real_volume - obj_volume)/obj_volume)*100:.1f}%'],
        ['RMSE', f'{exp_rmse:.3f} mm'],
        ['Overcut', f'{real_overcut:.2f} mm³'],
        ['Undercut', f'{real_undercut:.2f} mm³'],
        ['Correlation', f'{np.corrcoef(state[:,2], postScan_transformed[:,2])[0,1]:.3f}']
    ]
    
    # Create table
    table = ax.table(cellText=[[row[1]] for row in metrics_data],
                    rowLabels=[row[0] for row in metrics_data],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(metrics_data)):
        cell = table[(i, 0)]
        cell.set_facecolor('#f1f1f2')

    ax.set_title('6) Validation\nMetrics', fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# Skip statistical analysis figure for now (function not defined)

# Cross-section analysis removed as requested

#%% ========== SUMMARY METRICS AND FINAL REPORT ==========

def print_comprehensive_summary():
    """Print a comprehensive summary of all metrics"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VOLUMETRIC RESECTION ANALYSIS SUMMARY")
    print("="*80)
    
    # Basic volume metrics
    print(f"\n🎯 VOLUME METRICS:")
    print(f"   Objective Volume:     {obj_volume:.3f} mm³")
    print(f"   Simulation Volume:    {sim_volume:.3f} mm³")
    print(f"   Experimental Volume:  {real_volume:.3f} mm³")
    
    # Accuracy metrics
    sim_volume_error = abs(sim_volume - obj_volume) / obj_volume * 100
    exp_volume_error = abs(real_volume - obj_volume) / obj_volume * 100
    
    print(f"\n📊 VOLUME ACCURACY:")
    print(f"   Simulation Error:     {sim_volume_error:.2f}%")
    print(f"   Experimental Error:   {exp_volume_error:.2f}%")
    print(f"   Volume Difference:    {abs(real_volume - sim_volume):.3f} mm³")
    
    # Spatial accuracy
    sim_error = state[:,2] - objFxn[:,2]
    exp_error = postScan_transformed[:,2] - objFxn[:,2]
    
    sim_rmse = np.sqrt(np.mean(sim_error**2))
    exp_rmse = np.sqrt(np.mean(exp_error**2))
    sim_mae = np.mean(np.abs(sim_error))
    exp_mae = np.mean(np.abs(exp_error))
    
    print(f"\n📏 SPATIAL ACCURACY:")
    print(f"   Simulation RMSE:      {sim_rmse:.4f} mm")
    print(f"   Experimental RMSE:    {exp_rmse:.4f} mm")
    print(f"   Simulation MAE:       {sim_mae:.4f} mm")
    print(f"   Experimental MAE:     {exp_mae:.4f} mm")
    
    # Precision metrics
    print(f"\n🎯 PRECISION METRICS:")
    print(f"   Simulation Std:       {np.std(sim_error):.4f} mm")
    print(f"   Experimental Std:     {np.std(exp_error):.4f} mm")
    print(f"   Simulation Range:     {sim_error.max() - sim_error.min():.4f} mm")
    print(f"   Experimental Range:   {exp_error.max() - exp_error.min():.4f} mm")
    
    # Bias metrics
    print(f"\n📈 BIAS ANALYSIS:")
    print(f"   Simulation Bias:      {np.mean(sim_error):.4f} mm")
    print(f"   Experimental Bias:    {np.mean(exp_error):.4f} mm")
    
    # Overcut/Undercut analysis
    print(f"\n✂️  OVERCUT/UNDERCUT ANALYSIS:")
    print(f"   Simulation Overcut:   {sim_overcut:.3f} mm³")
    print(f"   Simulation Undercut:  {sim_undercut:.3f} mm³")
    print(f"   Experimental Overcut: {real_overcut:.3f} mm³")
    print(f"   Experimental Undercut:{real_undercut:.3f} mm³")

    print(f"\n✂️  OVER/UNDERCUT — as % of Objective volume:")
    print(f"   Simulation Overcut:   {pct(sim_overcut,  obj_volume):.1f}%")
    print(f"   Simulation Undercut:  {pct(sim_undercut, obj_volume):.1f}%")
    print(f"   Experimental Overcut: {pct(real_overcut, obj_volume):.1f}%")
    print(f"   Experimental Undercut:{pct(real_undercut, obj_volume):.1f}%")

    print(f"\n✂️  OVER/UNDERCUT — as % of Simulation volume:")
    print(f"   Simulation Overcut:   {pct(sim_overcut,  sim_volume):.1f}%")
    print(f"   Simulation Undercut:  {pct(sim_undercut, sim_volume):.1f}%")
    print(f"   Experimental Overcut: {pct(real_overcut, sim_volume):.1f}%")
    print(f"   Experimental Undercut:{pct(real_undercut, sim_volume):.1f}%")

    
    # Correlation analysis
    sim_obj_corr = np.corrcoef(state[:,2], objFxn[:,2])[0,1]
    exp_obj_corr = np.corrcoef(postScan_transformed[:,2], objFxn[:,2])[0,1]
    sim_exp_corr = np.corrcoef(state[:,2], postScan_transformed[:,2])[0,1]
    
    print(f"\n🔗 CORRELATION ANALYSIS:")
    print(f"   Sim-Objective:        {sim_obj_corr:.4f}")
    print(f"   Exp-Objective:        {exp_obj_corr:.4f}")
    print(f"   Sim-Experimental:     {sim_exp_corr:.4f}")
    
    # Statistical significance
    from scipy.stats import ttest_rel, wilcoxon
    
    try:
        t_stat, p_val_t = ttest_rel(sim_error, exp_error)
        w_stat, p_val_w = wilcoxon(sim_error, exp_error)
        
        print(f"\n📊 STATISTICAL TESTS:")
        print(f"   Paired t-test p-value:    {p_val_t:.6f}")
        print(f"   Wilcoxon p-value:         {p_val_w:.6f}")
        print(f"   Significant difference:   {'Yes' if p_val_t < 0.05 else 'No'} (α=0.05)")
    except:
        print(f"\n📊 STATISTICAL TESTS: Could not perform (insufficient data)")
    
    # Performance summary
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"   Overall Simulation Accuracy:    {100 - sim_volume_error:.1f}%")
    print(f"   Overall Experimental Accuracy:  {100 - exp_volume_error:.1f}%")
    print(f"   Spatial Agreement (Correlation): {sim_exp_corr:.3f}")
    print(f"   Prediction Quality:             {'Excellent' if sim_exp_corr > 0.9 else 'Good' if sim_exp_corr > 0.8 else 'Moderate'}")

    print(f"\n📊 ACHIEVEMENT RATIOS:")
    print(f"   Simulation vs Objective:  {100.0*sim_volume/obj_volume:.1f}%")
    print(f"   Experimental vs Objective:{100.0*real_volume/obj_volume:.1f}%")
    print(f"   Experimental vs Simulation:{100.0*real_volume/sim_volume:.1f}%")

    print(f"\n✂️  OVER/UNDERCUT (as % of Objective):")
    print(f"   Simulation Overcut:       {100.0*sim_overcut/obj_volume:.1f}%")
    print(f"   Simulation Undercut:      {100.0*sim_undercut/obj_volume:.1f}%")
    print(f"   Experimental Overcut:     {100.0*real_overcut/obj_volume:.1f}%")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if exp_rmse > sim_rmse * 1.5:
        print("   • Consider improving experimental setup or measurement accuracy")
    if abs(np.mean(exp_error)) > 0.1:
        print("   • Systematic bias detected - check calibration")
    if sim_exp_corr < 0.8:
        print("   • Low simulation-experimental correlation - review model parameters")
    if real_overcut > sim_overcut * 2:
        print("   • Experimental overcut significantly higher than predicted")
    
    print("\n" + "="*80)

# Print comprehensive summary
print_comprehensive_summary()

#%% ========== SAVE ALL ANALYSIS DATA ==========

def save_analysis_results():
    """Save all analysis results to files in the output folder"""

    # Use the same output folder as figures
    output_folder = 'test vol error'
    os.makedirs(output_folder, exist_ok=True)

    # Create analysis results dictionary
    analysis_results = {
        'volume_metrics': {
            'objective_volume': obj_volume,
            'simulation_volume': sim_volume,
            'experimental_volume': real_volume,
            'sim_volume_error_percent': abs(sim_volume - obj_volume) / obj_volume * 100,
            'exp_volume_error_percent': abs(real_volume - obj_volume) / obj_volume * 100
        },
        'spatial_metrics': {
            'sim_rmse': np.sqrt(np.mean((state[:,2] - objFxn[:,2])**2)),
            'exp_rmse': np.sqrt(np.mean((postScan_transformed[:,2] - objFxn[:,2])**2)),
            'sim_mae': np.mean(np.abs(state[:,2] - objFxn[:,2])),
            'exp_mae': np.mean(np.abs(postScan_transformed[:,2] - objFxn[:,2]))
        },
        'overcut_undercut': {
            'sim_overcut': sim_overcut,
            'sim_undercut': sim_undercut,
            'exp_overcut': real_overcut,
            'exp_undercut': real_undercut
        },
        'correlation_metrics': {
            'sim_obj_correlation': np.corrcoef(state[:,2], objFxn[:,2])[0,1],
            'exp_obj_correlation': np.corrcoef(postScan_transformed[:,2], objFxn[:,2])[0,1],
            'sim_exp_correlation': np.corrcoef(state[:,2], postScan_transformed[:,2])[0,1]
        },
        'error_statistics': {
            'sim_error_mean': np.mean(state[:,2] - objFxn[:,2]),
            'sim_error_std': np.std(state[:,2] - objFxn[:,2]),
            'exp_error_mean': np.mean(postScan_transformed[:,2] - objFxn[:,2]),
            'exp_error_std': np.std(postScan_transformed[:,2] - objFxn[:,2])
        }
    }

    # Save to JSON in output folder
    import json
    with open(os.path.join(output_folder, 'resection_analysis_results.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)

    # Save point clouds and errors for further analysis in output folder
    np.save(os.path.join(output_folder, 'objective_points.npy'), objFxn)
    np.save(os.path.join(output_folder, 'simulation_points.npy'), state)
    np.save(os.path.join(output_folder, 'experimental_points.npy'), postScan_transformed)
    np.save(os.path.join(output_folder, 'simulation_errors.npy'), state[:,2] - objFxn[:,2])
    np.save(os.path.join(output_folder, 'experimental_errors.npy'), postScan_transformed[:,2] - objFxn[:,2])

    print(f"\n💾 Analysis results saved in '{output_folder}' folder:")
    print("   • resection_analysis_results.json")
    print("   • objective_points.npy")
    print("   • simulation_points.npy")
    print("   • experimental_points.npy")
    print("   • simulation_errors.npy")
    print("   • experimental_errors.npy")

    print(f"\n📸 Figures saved in '{output_folder}' folder:")
    print("   • depth_maps_comparison.png/pdf")
    print("   • performance_metrics_table.png/pdf")
    print("   • volumetric_resection_analysis.png/pdf (comprehensive figure)")

    print(f"\n📊 Achievement Ratios:")
    print(f"   Experimental vs Objective: {100.0*real_volume/obj_volume:.1f}%")
    print(f"   Experimental Undercut: {100.0*real_undercut/obj_volume:.1f}%")


# Save all results
save_analysis_results()

print("\n🎉 ANALYSIS COMPLETE!")
print("All figures and data have been generated and saved.")
print("Ready for publication panel creation!")

plt.show()

#%% ========== 3D INTERACTIVE VISUALIZATION ==========

def create_interactive_3d_visualization():
    """Create an interactive 3D visualization using PyVista"""
    
    # Create a plotter with multiple viewports
    pl = pv.Plotter(shape=(2, 2), title='Interactive Volumetric Resection Analysis')
    
    # Subplot 1: Original point clouds
    pl.subplot(0, 0)
    pl.add_mesh(preScan_transformed, color=colors['objective'], point_size=3, 
                label='Pre-operative', opacity=0.8)
    pl.add_mesh(postScan_transformed, color=colors['experimental'], point_size=3, 
                label='Post-operative', opacity=0.8)
    pl.add_title('Original Point Clouds')
    pl.add_legend()
    
    # Subplot 2: Objective vs Results
    pl.subplot(0, 1)
    
    # Create mesh surfaces for better visualization
    def create_surface_from_points(points, resolution=50):
        """Create a surface mesh from point cloud"""
        x_range = np.linspace(points[:,0].min(), points[:,0].max(), resolution)
        y_range = np.linspace(points[:,1].min(), points[:,1].max(), resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Interpolate Z values
        Z = interp.griddata(points[:,0:2], points[:,2], (X, Y), method='cubic')
        
        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        return grid
    
    obj_surface = create_surface_from_points(objFxn)
    sim_surface = create_surface_from_points(state)
    exp_surface = create_surface_from_points(postScan_transformed)
    
    pl.add_mesh(obj_surface, color=colors['objective'], opacity=0.7, label='Objective')
    pl.add_mesh(sim_surface, color=colors['simulation'], opacity=0.6, label='Simulation')
    pl.add_mesh(exp_surface, color=colors['experimental'], opacity=0.5, label='Experimental')
    pl.add_title('Surface Comparison')
    pl.add_legend()
    
    # Subplot 3: Error visualization
    pl.subplot(1, 0)
    
    # Create error surface
    error_points = objFxn.copy()
    error_points[:,2] = postScan_transformed[:,2] - objFxn[:,2]
    
    # Color by error magnitude
    error_magnitude = np.abs(error_points[:,2])
    pl.add_mesh(error_points, scalars=error_points[:,2], point_size=5, 
                cmap='RdBu_r', clim=[-error_points[:,2].std()*2, error_points[:,2].std()*2])
    pl.add_title('Error Distribution')
    pl.add_scalar_bar('Error (mm)')
    
    # Subplot 4: Cross-sections
    pl.subplot(1, 1)
    
    # Create cross-section through center
    center_y = (objFxn[:,1].min() + objFxn[:,1].max()) / 2
    tolerance = 0.2
    
    # Find points near the center line
    center_mask = np.abs(objFxn[:,1] - center_y) < tolerance
    if np.sum(center_mask) > 0:
        center_obj = objFxn[center_mask]
        center_sim = state[center_mask]
        center_exp = postScan_transformed[center_mask]
        
        # Sort by X coordinate for line plot
        sort_idx = np.argsort(center_obj[:,0])
        
        # Create line plot using PyVista
        obj_line = pv.Spline(center_obj[sort_idx], n_points=100)
        sim_line = pv.Spline(center_sim[sort_idx], n_points=100)
        exp_line = pv.Spline(center_exp[sort_idx], n_points=100)
        
        pl.add_mesh(obj_line, color=colors['objective'], line_width=3, label='Objective')
        pl.add_mesh(sim_line, color=colors['simulation'], line_width=3, label='Simulation')
        pl.add_mesh(exp_line, color=colors['experimental'], line_width=3, label='Experimental')
    
    pl.add_title('Cross-section Profile')
    pl.add_legend()
    
    # Add camera positions for each subplot
    pl.subplot(0, 0)
    pl.camera.azimuth = 45
    pl.camera.elevation = 30
    
    pl.subplot(0, 1)
    pl.camera.azimuth = 45
    pl.camera.elevation = 30
    
    pl.subplot(1, 0)
    pl.camera.azimuth = 45
    pl.camera.elevation = 30
    
    pl.subplot(1, 1)
    pl.camera.azimuth = 0
    pl.camera.elevation = 0
    
    return pl

# Create interactive visualization (note: this will open an interactive window)
try:
    interactive_pl = create_interactive_3d_visualization()
    # Save screenshots
    interactive_pl.screenshot('3d_interactive_view.png', window_size=[1600, 1200])
    interactive_pl.show()
except:
    print("Interactive 3D visualization requires display. Skipping...")

#%% ========== PUBLICATION-READY INDIVIDUAL FIGURES ==========

def create_depth_maps_figure():
    """
    Figure 1: Objective, Simulation, and Experimental depth maps with gentle colormap
    """
    # Build a common grid
    x_range = np.linspace(objFxn[:,0].min(), objFxn[:,0].max(), 140)
    y_range = np.linspace(objFxn[:,1].min(), objFxn[:,1].max(), 140)
    X, Y = np.meshgrid(x_range, y_range)

    # Interpolated depth fields on the same grid
    obj_grid = interp.griddata(objFxn[:,0:2], objFxn[:,2], (X, Y), method='cubic')
    sim_grid = interp.griddata(state[:,0:2], state[:,2], (X, Y), method='cubic')
    exp_grid = interp.griddata(postScan_transformed[:,0:2], postScan_transformed[:,2], (X, Y), method='cubic')

    # Unified depth limits
    dmin, dmax = unified_depth_limits(objFxn, state, postScan_transformed)

    # Figure layout - single row with gentle colormap
    fig = plt.figure(figsize=(15, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1,1,1,0.08], wspace=0.3)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1]); ax3 = fig.add_subplot(gs[2]); cax = fig.add_subplot(gs[3])

    # Use gentle colormap instead of RdYlBu_r
    gentle_cmap = 'viridis'  # You can also use 'plasma', 'cividis', or 'Blues'

    # Objective
    im1 = ax1.contourf(X, Y, obj_grid, levels=20, cmap=gentle_cmap, vmin=dmin, vmax=dmax)
    ax1.contour(X, Y, obj_grid, levels=10, colors='white', alpha=0.4, linewidths=0.6)
    ax1.set_title('Objective', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_aspect('equal')

    # Simulation
    im2 = ax2.contourf(X, Y, sim_grid, levels=20, cmap=gentle_cmap, vmin=dmin, vmax=dmax)
    ax2.contour(X, Y, sim_grid, levels=10, colors='white', alpha=0.4, linewidths=0.6)
    ax2.set_title('Simulation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_aspect('equal')

    # Experimental
    im3 = ax3.contourf(X, Y, exp_grid, levels=20, cmap=gentle_cmap, vmin=dmin, vmax=dmax)
    ax3.contour(X, Y, exp_grid, levels=10, colors='white', alpha=0.4, linewidths=0.6)
    ax3.set_title('Experimental', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Y (mm)'); ax3.set_aspect('equal')

    # Shared colorbar
    cbar = plt.colorbar(im3, cax=cax)
    cbar.set_label('Depth (mm)', fontweight='bold')

    fig.tight_layout()
    return fig

# Volume achievement analysis removed as requested

def create_key_metrics_figure():
    """
    Figure 3: Key performance metrics table
    """
    # Calculate all metrics
    sim_over_pct_obj = pct(sim_overcut, obj_volume)
    sim_under_pct_obj = pct(sim_undercut, obj_volume)
    exp_over_pct_obj = pct(real_overcut, obj_volume)
    exp_under_pct_obj = pct(real_undercut, obj_volume)

    sim_over_pct_sim = pct(sim_overcut, sim_volume)
    sim_under_pct_sim = pct(sim_undercut, sim_volume)
    exp_over_pct_sim = pct(real_overcut, sim_volume)
    exp_under_pct_sim = pct(real_undercut, sim_volume)

    sim_rmse = np.sqrt(np.mean((state[:,2] - objFxn[:,2])**2))
    exp_rmse = np.sqrt(np.mean((postScan_transformed[:,2] - objFxn[:,2])**2))
    sim_mae = np.mean(np.abs(state[:,2] - objFxn[:,2]))
    exp_mae = np.mean(np.abs(postScan_transformed[:,2] - objFxn[:,2]))

    correlation = np.corrcoef(state[:,2], postScan_transformed[:,2])[0,1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Create comprehensive metrics table
    rows = [
        ['Metric', 'Simulation', 'Experimental', 'Notes'],
        ['Overcut',
         f'{sim_over_pct_obj:.1f}% of Obj\n{sim_over_pct_sim:.1f}% of Sim',
         f'{exp_over_pct_obj:.1f}% of Obj\n{exp_over_pct_sim:.1f}% of Sim',
         'Excess removal beyond objective'],
        ['Undercut',
         f'{sim_under_pct_obj:.1f}% of Obj\n{sim_under_pct_sim:.1f}% of Sim',
         f'{exp_under_pct_obj:.1f}% of Obj\n{exp_under_pct_sim:.1f}% of Sim',
         'Insufficient removal vs objective'],
        ['RMSE (mm)', f'{sim_rmse:.3f}', f'{exp_rmse:.3f}', 'Root mean square error'],
        ['MAE (mm)', f'{sim_mae:.3f}', f'{exp_mae:.3f}', 'Mean absolute error'],
        ['Volume Accuracy',
         f'{(1-abs(sim_volume-obj_volume)/obj_volume)*100:.1f}%',
         f'{(1-abs(real_volume-obj_volume)/obj_volume)*100:.1f}%',
         'Volume achievement accuracy'],
        ['Correlation vs Objective',
         f'{np.corrcoef(state[:,2], objFxn[:,2])[0,1]:.3f}',
         f'{np.corrcoef(postScan_transformed[:,2], objFxn[:,2])[0,1]:.3f}',
         'Spatial correlation with target'],
        ['Sim-Exp Correlation', f'{correlation:.3f}', '-', 'Agreement between sim and exp'],
    ]

    # Create table
    table = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style the table
    for i in range(len(rows[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#f1f1f2')
        cell.set_text_props(weight='bold')

    fig.tight_layout()
    return fig

# Create output folder
import os
output_folder = 'test vol error'
os.makedirs(output_folder, exist_ok=True)

# Create remaining figures
depth_fig = create_depth_maps_figure()
metrics_fig = create_key_metrics_figure()

# Save figures with appropriate names in the output folder
depth_fig.savefig(os.path.join(output_folder, 'depth_maps_comparison.png'), dpi=300, bbox_inches='tight')
depth_fig.savefig(os.path.join(output_folder, 'depth_maps_comparison.pdf'), bbox_inches='tight')

metrics_fig.savefig(os.path.join(output_folder, 'performance_metrics_table.png'), dpi=300, bbox_inches='tight')
metrics_fig.savefig(os.path.join(output_folder, 'performance_metrics_table.pdf'), bbox_inches='tight')

plt.show()

#%% ========== STATISTICAL ANALYSIS FIGURE ==========

def create_statistical_analysis_figure():
    """Detailed statistical analysis figure"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Statistical Analysis of Volumetric Resection Performance', fontsize=18, fontweight='bold')
    
    # Calculate errors
    sim_error = state[:,2] - objFxn[:,2]
    exp_error = postScan_transformed[:,2] - objFxn[:,2]
    
    # A) Error distributions
    ax_a = fig.add_subplot(gs[0, 0:2])
    
    bins = np.linspace(min(sim_error.min(), exp_error.min()), 
                      max(sim_error.max(), exp_error.max()), 30)
    
    ax_a.hist(sim_error, bins=bins, alpha=0.7, label='Simulation Error', 
             color=colors['simulation'], density=True, edgecolor='black')
    ax_a.hist(exp_error, bins=bins, alpha=0.7, label='Experimental Error', 
             color=colors['experimental'], density=True, edgecolor='black')
    
    # Add normal distribution fits
    from scipy.stats import norm
    sim_mu, sim_std = norm.fit(sim_error)
    exp_mu, exp_std = norm.fit(exp_error)
    
    x_fit = np.linspace(bins[0], bins[-1], 100)
    ax_a.plot(x_fit, norm.pdf(x_fit, sim_mu, sim_std), '--', 
             color=colors['simulation'], linewidth=2, label='Sim Normal Fit')
    ax_a.plot(x_fit, norm.pdf(x_fit, exp_mu, exp_std), '--', 
             color=colors['experimental'], linewidth=2, label='Exp Normal Fit')
    
    ax_a.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax_a.set_xlabel('Error (mm)', fontweight='bold')
    ax_a.set_ylabel('Density', fontweight='bold')
    ax_a.set_title('A) Error Distribution Analysis', fontweight='bold')
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Simulation: μ={sim_mu:.3f}, σ={sim_std:.3f}\nExperimental: μ={exp_mu:.3f}, σ={exp_std:.3f}'
    ax_a.text(0.02, 0.98, stats_text, transform=ax_a.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # B) Box plots
    ax_b = fig.add_subplot(gs[0, 2])
    
    box_data = [sim_error, exp_error]
    box_labels = ['Simulation', 'Experimental']
    
    bp = ax_b.boxplot(box_data, labels=box_labels, patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor(colors['simulation'])
    bp['boxes'][1].set_facecolor(colors['experimental'])
    
    ax_b.set_ylabel('Error (mm)', fontweight='bold')
    ax_b.set_title('B) Error Distribution\nBoxplots', fontweight='bold')
    ax_b.grid(True, alpha=0.3)
    ax_b.axhline(0, color='red', linestyle='--', alpha=0.7)
    
    # C) Bland-Altman plot
    ax_c = fig.add_subplot(gs[0, 3])
    
    diff = state[:,2] - postScan_transformed[:,2]
    mean_val = (state[:,2] + postScan_transformed[:,2]) / 2
    
    ax_c.scatter(mean_val, diff, alpha=0.6, s=15, color=colors['simulation'])
    
    # Calculate limits of agreement
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    ax_c.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.3f}')
    ax_c.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=1.5, 
                label=f'+1.96SD: {mean_diff + 1.96*std_diff:.3f}')
    ax_c.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=1.5, 
                label=f'-1.96SD: {mean_diff - 1.96*std_diff:.3f}')
    
    ax_c.set_xlabel('Mean Depth (mm)', fontweight='bold')
    ax_c.set_ylabel('Difference (mm)', fontweight='bold')
    ax_c.set_title('C) Bland-Altman Plot', fontweight='bold')
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)
    
    # D) Correlation analysis
    ax_d = fig.add_subplot(gs[1, 0])
    
    # Scatter plot with density
    scatter = ax_d.scatter(state[:,2], postScan_transformed[:,2], 
                          c=objFxn[:,2], cmap='viridis', alpha=0.6, s=20)
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(state[:,2], postScan_transformed[:,2])
    
    x_reg = np.array([state[:,2].min(), state[:,2].max()])
    y_reg = slope * x_reg + intercept
    
    ax_d.plot(x_reg, y_reg, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')
    
    # Add perfect correlation line
    min_val = min(state[:,2].min(), postScan_transformed[:,2].min())
    max_val = max(state[:,2].max(), postScan_transformed[:,2].max())
    ax_d.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect')
    
    ax_d.set_xlabel('Simulation Depth (mm)', fontweight='bold')
    ax_d.set_ylabel('Experimental Depth (mm)', fontweight='bold')
    ax_d.set_title('D) Correlation Analysis', fontweight='bold')
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'r = {r_value:.3f}\np = {p_value:.2e}\nRMSE = {np.sqrt(np.mean(diff**2)):.3f}'
    ax_d.text(0.05, 0.95, stats_text, transform=ax_d.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(scatter, ax=ax_d, label='Objective Depth (mm)')
    
    # E) Residuals vs fitted
    ax_e = fig.add_subplot(gs[1, 1])
    
    fitted_values = slope * state[:,2] + intercept
    residuals = postScan_transformed[:,2] - fitted_values
    
    ax_e.scatter(fitted_values, residuals, alpha=0.6, s=15, color=colors['experimental'])
    ax_e.axhline(0, color='red', linestyle='--', linewidth=2)
    
    ax_e.set_xlabel('Fitted Values', fontweight='bold')
    ax_e.set_ylabel('Residuals', fontweight='bold')
    ax_e.set_title('E) Residuals vs Fitted', fontweight='bold')
    ax_e.grid(True, alpha=0.3)
    
    # F) Q-Q plot
    ax_f = fig.add_subplot(gs[1, 2])
    
    from scipy import stats
    stats.probplot(exp_error, dist="norm", plot=ax_f)
    ax_f.set_title('F) Q-Q Plot\n(Experimental Error)', fontweight='bold')
    ax_f.grid(True, alpha=0.3)
    
    # G) Performance radar chart
    ax_g = fig.add_subplot(gs[1, 3], projection='polar')
    
    categories = ['Volume\nAccuracy', 'RMSE\nAccuracy', 'Precision', 'Bias', 'Correlation']
    
    # Calculate normalized performance metrics (0-100 scale)
    volume_acc = (1 - abs(real_volume - obj_volume)/obj_volume) * 100
    rmse_acc = max(0, 100 - np.sqrt(np.mean(exp_error**2)) * 50)  # Scaled
    precision = max(0, 100 - np.std(exp_error) * 50)  # Scaled
    bias = max(0, 100 - abs(np.mean(exp_error)) * 50)  # Scaled
    correlation = abs(r_value) * 100
    
    values = [volume_acc, rmse_acc, precision, bias, correlation]
    
    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    ax_g.plot(angles, values, 'o-', linewidth=2, color=colors['experimental'])
    ax_g.fill(angles, values, alpha=0.25, color=colors['experimental'])
    ax_g.set_xticks(angles[:-1])
    ax_g.set_xticklabels(categories)
    ax_g.set_ylim(0, 100)
    ax_g.set_title('G) Performance Radar', fontweight='bold', pad=20)
    ax_g.grid(True)
    
    # H) Error spatial distribution
    ax_h = fig.add_subplot(gs[2, 0:2])
    
    # Create 2D histogram of errors
    H, xedges, yedges = np.histogram2d(objFxn[:,0], objFxn[:,1], bins=20, weights=exp_error)
    H_counts, _, _ = np.histogram2d(objFxn[:,0], objFxn[:,1], bins=20)
    
    # Avoid division by zero
    H_avg = np.divide(H, H_counts, out=np.zeros_like(H), where=H_counts!=0)
    
    im = ax_h.imshow(H_avg.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     origin='lower', cmap='RdBu_r', aspect='equal')
    
    ax_h.set_xlabel('X Position (mm)', fontweight='bold')
    ax_h.set_ylabel('Y Position (mm)', fontweight='bold')
    ax_h.set_title('H) Spatial Error Distribution', fontweight='bold')
    
    divider = make_axes_locatable(ax_h)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Average Error (mm)')
    
    # I) Statistical summary table
    ax_i = fig.add_subplot(gs[2, 2:])
    ax_i.axis('off')
    
    # Create comprehensive statistics table
    stats_data = [
        ['Metric', 'Simulation', 'Experimental', 'Difference'],
        ['Mean Error (mm)', f'{np.mean(sim_error):.4f}', f'{np.mean(exp_error):.4f}', f'{np.mean(exp_error) - np.mean(sim_error):.4f}'],
        ['Std Error (mm)', f'{np.std(sim_error):.4f}', f'{np.std(exp_error):.4f}', f'{np.std(exp_error) - np.std(sim_error):.4f}'],
        ['RMSE (mm)', f'{np.sqrt(np.mean(sim_error**2)):.4f}', f'{np.sqrt(np.mean(exp_error**2)):.4f}', 
         f'{np.sqrt(np.mean(exp_error**2)) - np.sqrt(np.mean(sim_error**2)):.4f}'],
        ['MAE (mm)', f'{np.mean(np.abs(sim_error)):.4f}', f'{np.mean(np.abs(exp_error)):.4f}', 
         f'{np.mean(np.abs(exp_error)) - np.mean(np.abs(sim_error)):.4f}'],
        ['Volume (mm³)', f'{sim_volume:.3f}', f'{real_volume:.3f}', f'{real_volume - sim_volume:.3f}'],
        ['Volume Accuracy (%)', f'{(1-abs(sim_volume-obj_volume)/obj_volume)*100:.2f}', 
         f'{(1-abs(real_volume-obj_volume)/obj_volume)*100:.2f}', ''],
        ['Correlation with Obj', f'{np.corrcoef(state[:,2], objFxn[:,2])[0,1]:.4f}', 
         f'{np.corrcoef(postScan_transformed[:,2], objFxn[:,2])[0,1]:.4f}', ''],
        ['Sim-Exp Correlation', '', f'{r_value:.4f}', f'p = {p_value:.2e}']
    ]
    
    # Create table
    table = ax_i.table(cellText=stats_data[1:], colLabels=stats_data[0],
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)