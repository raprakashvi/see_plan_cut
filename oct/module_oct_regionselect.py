import numpy as np
import open3d as o3d
import sys
import time
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import pandas as pd
from scipy.stats import trim_mean
import os
import scipy.optimize as opt
import cv2
import csv
from typing import Tuple, List



def load_pcd(path, visualize=True, source="oct"):
    pcd = o3d.io.read_point_cloud(path)
    print(f"Loaded {len(pcd.points)} points from {path}")

    if source.lower().startswith("realsense"):
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # 1) fix axes
        R_flip = R.from_euler('x', 180, degrees=True).as_matrix()
        R_tilt = R.from_euler('y', -90,  degrees=True).as_matrix()
        T = np.eye(4)
        T[:3,:3] = R_tilt @ R_flip
        pcd.transform(T)

    # 2) convert units if needed
    # pcd.points = Vector3dVector(np.asarray(pcd.points) * 1e-3)

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        # point the normals upward in the new frame
        max_z = np.max(np.asarray(pcd.points)[:,2])
        pcd.orient_normals_towards_camera_location(camera_location=[0,0,max_z+0.1])

        # ensure purely upward normals
        normals = np.asarray(pcd.normals)
        neg = normals[:,2] < 0
        normals[neg] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)

    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    print(f"Cloud spans (x,y,z): ({min_bound[0]:.4f}, {min_bound[1]:.4f}, {min_bound[2]:.4f}) to ({max_bound[0]:.4f}, {max_bound[1]:.4f}, {max_bound[2]:.4f})")



    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name="Aligned PointCloud")

    return pcd



def select_region_mode():
    """Let user choose selection mode"""
    while True:
        print("\nSelect mode:")
        print("1. Point and radius")
        print("2. Rectangle region")
        print("3. Square region around point (with consistent waypoints)")
        print("4. Laser center targets detection")
        choice = input("Enter choice (1, 2, 3, 4): ")
        if choice in ['1', '2', '3','4']:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, 3, 4.")

def select_point_region(pcd):
    """Select a point and specify radius"""
    # Create visualizer for point selection
    print("Please pick a point. Press 'Q' to exit selection mode.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    
    # Get picked points
    picked_points = vis.get_picked_points()
    if not picked_points:
        print("No point selected. Exiting.")
        return None
    
    # Get selected point coordinates
    center_point = np.asarray(pcd.points)[picked_points[0]]
    
    # Get radius from user
    while True:
        try:
            radius = float(input("Enter radius around point (in meters): "))
            if radius > 0:
                break
            print("Radius must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Find points within radius
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points - center_point, axis=1)
    print("Min and max values of original pointcloud")
    print(f"X: {np.min(points[:, 0]):.4f}, {np.max(points[:, 0]):.4f}")
    print(f"Selected point: {center_point}")
    print(f"Radius: {radius}")
    print(f"Number of points within radius: {np.sum(distances <= radius)}")
    mask = distances <= radius
    print(f"Min and max values of selected region")
    print(f"X: {np.min(points[mask, 0]):.4f}, {np.max(points[mask, 0]):.4f}")
    
    # Create new point cloud with selected points
    selected_pcd = o3d.geometry.PointCloud()
    selected_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_normals():
        selected_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    if len(np.asarray(pcd.colors)) > 0:
        selected_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    
    return selected_pcd

def select_rectangle_region(pcd):
    """Select a rectangular region using two corner points"""
    print("Please pick two points to define the rectangle corners. Press 'Q' after selecting.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    
    # Get picked points
    picked_points = vis.get_picked_points()
    if len(picked_points) < 2:
        print("Need two points to define rectangle. Exiting.")
        return None
    
    # Get coordinates of the two points
    points = np.asarray(pcd.points)
    p1 = points[picked_points[0]]
    p2 = points[picked_points[1]]

    print("Max and min values of original pointcloud")
    print(f"X: {np.min(points[:, 0]):.4f}, {np.max(points[:, 0]):.4f}")
    print(f"Y: {np.min(points[:, 1]):.4f}, {np.max(points[:, 1]):.4f}")
    print(f"Z: {np.min(points[:, 2]):.4f}, {np.max(points[:, 2]):.4f}")
    
    # Create bounds (only for x and y coordinates)
    min_x = min(p1[0], p2[0])
    max_x = max(p1[0], p2[0])
    min_y = min(p1[1], p2[1])
    max_y = max(p1[1], p2[1])
    
    # Find points within x and y bounds (ignoring z)
    mask = (points[:, 0] >= min_x) & (points[:, 0] <= max_x) & \
           (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    # Create new point cloud with selected points
    selected_pcd = o3d.geometry.PointCloud()
    selected_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_normals():
        selected_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    if len(np.asarray(pcd.colors)) > 0:
        selected_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    
    return selected_pcd

def pick_point(pcd):
    """Let user interactively pick a point"""
    print("Please pick a seed point. Press 'Q' to exit selection mode.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    
    picked_points = vis.get_picked_points()
    if not picked_points:
        print("No point selected. Exiting.")
        sys.exit(1)
    
    return np.asarray(pcd.points)[picked_points[0]]

def select_square_region(
    pcd,
    seed_point,
    region_size: float = 0.03,
    normal_radius: float = 0.003,
    normal_max_nn: int = 10,
    orient_consistent: bool = False
):
    """Select a square region with improved coordinate system detection"""
    print("Processing square region...")
    half = region_size / 2
    pts = np.asarray(pcd.points)
    mask = (
        (np.abs(pts[:, 0] - seed_point[0]) <= half) &
        (np.abs(pts[:, 1] - seed_point[1]) <= half)
    )
    region_pts = pts[mask]
    print(f"Found {len(region_pts)} points in region")
    
    region_pcd = o3d.geometry.PointCloud()
    region_pcd.points = o3d.utility.Vector3dVector(region_pts)
    
    # Estimate normals with smaller parameters for better local accuracy
    print("Computing normals...")
    region_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    
    # Optional consistent orientation
    if orient_consistent:
        region_pcd.orient_normals_consistent_tangent_plane(k=15)
    
    normals = np.asarray(region_pcd.normals)
    
    # Determine overall coordinate system orientation
    # Method 1: Use dominant normal direction
    avg_normal_z = np.mean(normals[:, 2])
    invert_system = avg_normal_z < 0
    
    # Method 2: Check if there's a clear plane and its orientation
    # For surface scanning, we'd expect most normals to point in similar direction
    if abs(avg_normal_z) < 0.3:  # If average normal is too horizontal
        # Try to find a more robust orientation by analyzing the histogram of z components
        z_hist, _ = np.histogram(normals[:, 2], bins=20, range=(-1, 1))
        negative_count = np.sum(z_hist[:10])
        positive_count = np.sum(z_hist[10:])
        invert_system = negative_count > positive_count

    # # Show the calculated inversion and use it as default
    # default_response = "y" if invert_system else "n"
    # user_response = input(
    #     f"Average normal Z: {avg_normal_z:.4f}. Invert coordinate system? [Y/n]: " if invert_system else
    #     f"Average normal Z: {avg_normal_z:.4f}. Invert coordinate system? [y/N]: "
    # ).strip().lower()
    
    # # Use precalculated default if user just presses Enter
    # if user_response == "":
    #     user_response = default_response
    
    # invert_system = user_response == 'y'
    
    # if invert_system:
    #     print("Inverting coordinate system based on calculation/user input")
    # else:
    #     print("Keeping original coordinate system")
    
    print(f"Average normal Z: {avg_normal_z:.4f}")
    print(f"Coordinate system needs inversion: {invert_system}")
    
    # Apply consistent orientation
    if invert_system:
        print("Inverting normals to maintain consistent orientation")
        normals = -normals
    
    # Force all normals to point in dominant direction
    # If inverted, we force downward (-Z), else upward (+Z)
    correct_dir = -1 if invert_system else 1
    flip_mask = (normals[:, 2] * correct_dir) < 0
    normals[flip_mask] = -normals[flip_mask]
    
    region_pcd.normals = o3d.utility.Vector3dVector(normals)
    region_pcd.paint_uniform_color([0, 1, 0])
    
    print("Region processing complete")
    return region_pts, normals, region_pcd


def compute_grid_origin(region_points: np.ndarray) -> np.ndarray:
    """Return the (xmin,ymin) of the patch, in XY."""
    return np.min(region_points[:, :2], axis=0)


def generate_grid_waypoints(
    region_points: np.ndarray,
    region_normals: np.ndarray,
    cell_size: float = 0.01,
    height_offset: float = 0.03,
    z_std_filter: float = 1.5
):
    """
    Tile the (x,y) plane into square cells of `cell_size`, then for each cell:
      - discard if <10 pts
      - robustly compute (x,y) median and z-trimmed mean
      - apply height_offset along the averaged normal vector
      - filter outliers by z_std_filter×std before recomputing
      - weight‐avg normals by distance to median
      - produce (position, normal, quaternion)

    Returns:
        waypoints: list of (position, normal, quaternion)
        stats_df: pandas DataFrame of per-cell stats
        origin: (xmin, ymin) of the grid
    """
    xy = region_points[:, :2]
    origin = xy.min(axis=0)

    x_idx = np.floor((xy[:, 0] - origin[0]) / cell_size).astype(int)
    y_idx = np.floor((xy[:, 1] - origin[1]) / cell_size).astype(int)
    unique_cells, inv = np.unique(np.vstack((x_idx, y_idx)).T, axis=0, return_inverse=True)

    waypoints, stats = [], []
    for cid, (cx, cy) in enumerate(unique_cells):
        idxs = np.where(inv == cid)[0]
        if idxs.size < 10:
            continue

        pts = region_points[idxs]
        nms = region_normals[idxs]
        z_vals = pts[:, 2]

        raw_std = np.std(z_vals)
        if z_std_filter > 0:
            mask = np.abs(z_vals - np.median(z_vals)) <= z_std_filter * raw_std
            if mask.sum() < 5:
                mask = np.ones_like(mask, dtype=bool)
            pts, nms, z_vals = pts[mask], nms[mask], pts[mask][:,2]

        avg_x, avg_y = np.median(pts[:, :2], axis=0)
        base_z = trim_mean(z_vals, 0.1)
        avg_point = np.array([avg_x, avg_y, base_z])

        d2 = np.sum((pts[:, :2] - avg_point[:2])**2, axis=1)
        w = (1.0 / (1.0 + d2))
        w /= w.sum()
        avg_n = (w[:, None] * nms).sum(axis=0)
        avg_n /= np.linalg.norm(avg_n)
        if avg_n[2] < 0:
            avg_n = -avg_n

        position = avg_point + height_offset * avg_n
        quat = normal_to_quaternion(avg_n)
        waypoints.append((position, avg_n, quat))

        angles = np.arccos(np.clip(nms.dot(avg_n), -1.0, 1.0)) * 180/np.pi
        stats.append({
            'cell_x': cx, 'cell_y': cy,
            'points_count': idxs.size, 'filtered_points': pts.shape[0],
            'avg_x': avg_x, 'avg_y': avg_y, 'base_z': base_z,
            'min_z': z_vals.min(), 'max_z': z_vals.max(), 'std_z': z_vals.std(),
            'normal_x': avg_n[0], 'normal_y': avg_n[1], 'normal_z': avg_n[2],
            'normal_std_angle': angles.std(), 'normal_max_angle': angles.max()
        })

    return waypoints, pd.DataFrame(stats), origin

def normal_to_quaternion(normal, invert_system=False):
    """Convert normal vector to quaternion respecting coordinate system orientation"""
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Choose reference vector based on coordinate system orientation
    if invert_system:
        ref = np.array([0, 0, -1])  # Use negative Z as reference if inverted
    else:
        ref = np.array([0, 0, 1])   # Use positive Z as reference otherwise
    
    # If normal is nearly parallel to ref, use a different reference
    if np.abs(np.dot(normal, ref)) > 0.99:
        ref = np.array([1, 0, 0])
    
    # Create orthogonal basis
    # Y-axis is along the normal
    y = normal
    
    # X-axis perpendicular to both normal and reference
    x = np.cross(ref, y)
    x = x / np.linalg.norm(x)
    
    # Z-axis completes right-handed basis
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    
    # Form rotation matrix and convert to quaternion
    R_matrix = np.column_stack((x, y, z))
    return R.from_matrix(R_matrix).as_quat()

# 3) rewrite add_cell_grid to take the same origin
def add_cell_grid(
    vis,
    region_points: np.ndarray,
    origin: np.ndarray,
    cell_size: float
):
    """
    Draw exactly the same grid used in generate_grid_waypoints:
    - origin is (xmin,ymin) of your patch
    - cell_size is the tiling pitch
    """
    # 1) figure out how many cells in each direction
    xy = region_points[:, :2]
    max_xy = np.max(xy, axis=0)
    nx = int(np.ceil((max_xy[0] - origin[0]) / cell_size))
    ny = int(np.ceil((max_xy[1] - origin[1]) / cell_size))

    z_plane = float(np.median(region_points[:,2]))
    grid_pts, grid_lines = [], []
    idx = 0

    # vertical lines
    for i in range(nx+1):
        x = origin[0] + i*cell_size
        for j in range(ny):
            y0 = origin[1] + j*cell_size
            y1 = y0 + cell_size
            grid_pts.append([x,y0,z_plane])
            grid_pts.append([x,y1,z_plane])
            grid_lines.append([idx,idx+1])
            idx += 2

    # horizontal lines
    for j in range(ny+1):
        y = origin[1] + j*cell_size
        for i in range(nx):
            x0 = origin[0] + i*cell_size
            x1 = x0 + cell_size
            grid_pts.append([x0,y,z_plane])
            grid_pts.append([x1,y,z_plane])
            grid_lines.append([idx,idx+1])
            idx += 2

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(grid_pts)),
        lines=o3d.utility.Vector2iVector(np.array(grid_lines))
    )
    ls.colors = o3d.utility.Vector3dVector(
        np.tile([0.7,0.7,0.7], (len(grid_lines),1))
    )
    vis.add_geometry(ls)

def visualize_waypoints(
    pcd: o3d.geometry.PointCloud,
    waypoints: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    normal_scale: float = 0.01,
    show_frames: bool = False
):
    """
    Draw:
      - the full cloud (pcd),
      - all waypoints as merged red spheres,
      - and normals as blue lines.
    (Grid visualization removed.)
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Waypoints")
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    vis.add_geometry(pcd)

    mesh_spheres = o3d.geometry.TriangleMesh()
    all_pts, all_lines, all_colors = [], [], []
    idx = 0

    for pos, normal, quat in waypoints:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=normal_scale * 0.2)
        sph.translate(pos)
        sph.paint_uniform_color([1, 0, 0])
        mesh_spheres += sph

        p1, p2 = pos, pos + normal_scale * normal
        all_pts.extend([p1, p2])
        all_lines.append([idx, idx + 1])
        all_colors.append([0, 0, 1])
        idx += 2

        if show_frames:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=normal_scale * 2)
            Rm = R.from_quat(quat).as_matrix()
            frame.rotate(Rm, center=(0, 0, 0))
            frame.translate(pos)
            vis.add_geometry(frame)

    vis.add_geometry(mesh_spheres)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(all_pts)),
        lines=o3d.utility.Vector2iVector(np.array(all_lines))
    )
    ls.colors = o3d.utility.Vector3dVector(np.array(all_colors))
    vis.add_geometry(ls)

    vis.run()
    vis.destroy_window()

def save_waypoints(waypoints, filename="waypoints.txt"):
    """Save waypoints to file for later use"""
    with open(filename, 'w') as f:
        f.write(f"{len(waypoints)}\n")
        for position, normal, quaternion in waypoints:
            # Position
            f.write(f"{position[0]} {position[1]} {position[2]} ")
            # Normal
            f.write(f"{normal[0]} {normal[1]} {normal[2]} ")
            # Quaternion
            f.write(f"{quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}\n")
    print(f"Saved {len(waypoints)} waypoints to {filename}")

def save_and_visualize(pcd, filename="selected_region.pcd"):
    """Save and visualize the selected region"""
    if pcd is None:
        return
    
    selected_points = np.asarray(pcd.points)
    print("Min and max values of selected region")
    print(f"X: {np.min(selected_points[:, 0]):.4f}, {np.max(selected_points[:, 0]):.4f}")
    print(f"Y: {np.min(selected_points[:, 1]):.4f}, {np.max(selected_points[:, 1]):.4f}")
    print(f"Z: {np.min(selected_points[:, 2]):.4f}, {np.max(selected_points[:, 2]):.4f}")
    
    # Save the point cloud
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Selected region saved as '{filename}'")
    
    # Visualize the result
    o3d.visualization.draw_geometries([pcd], window_name="Selected Region")


def find_laser_centers(pcd=None,npy_path=None, rgb_path=None):
    OUTDIR   = os.path.join("./laser_centers")
    os.makedirs(OUTDIR, exist_ok=True)

    GRID = 512           # XY raster size
    EXPECTED_N = 9       # keep at most this many blobs
    # TODO: Increase this value for detecting bigger dots
    ROW_BG_K =202       # odd; bigger -> stronger stripe removal

    # ---------- Load ----------
    if pcd is None:
        pts  = np.load(npy_path)
        cols = np.load(rgb_path)
        print(f"Loaded {len(pts)} points from {npy_path} and {rgb_path}")
    else:
        pcd = o3d.io.read_point_cloud(pcd)
        pts  = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        print(f"Loaded {len(pts)} points from {pcd}")
    x, y, z = pts[:,0], pts[:,1], pts[:,2]

    # RGB or gray -> uint8 intensity
    if cols.ndim == 2 and cols.shape[1] == 3:
        c = cols.astype(np.float32)
        if c.max() <= 1.0: c *= 255.0
        gray = (0.299*c[:,0] + 0.587*c[:,1] + 0.114*c[:,2]).astype(np.uint8)
    elif cols.ndim == 1:
        c = cols.astype(np.float32)
        if c.max() <= 1.0: c *= 255.0
        gray = c.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected cols shape {cols.shape}")

    # ---------- Map to XY grid ----------
    xmin, xmax = np.percentile(x, [0.5, 99.5]); xr = xmax - xmin
    ymin, ymax = np.percentile(y, [0.5, 99.5]); yr = ymax - ymin
    mx, my = 0.02*xr, 0.02*yr
    xmin -= mx; xmax += mx; ymin -= my; ymax += my
    sx = (GRID-1)/(xmax-xmin + 1e-12)
    sy = (GRID-1)/(ymax-ymin + 1e-12)
    ix = np.clip(((x - xmin)*sx).astype(np.int32), 0, GRID-1)
    iy = np.clip(((y - ymin)*sy).astype(np.int32), 0, GRID-1)

    # ---------- Mean aggregator (no banding) ----------
    sum_img = np.zeros((GRID, GRID), np.float32)
    cnt_img = np.zeros((GRID, GRID), np.int32)
    np.add.at(sum_img, (iy, ix), gray.astype(np.float32))
    np.add.at(cnt_img, (iy, ix), 1)
    mean_img = sum_img / np.maximum(cnt_img, 1)

    # Inpaint empty cells (cnt==0) for visual continuity
    mean_u8 = cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    holes = ((cnt_img == 0).astype(np.uint8))*255
    mean_u8 = cv2.inpaint(mean_u8, holes, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.join(OUTDIR, "01_enface_mean.png"), mean_u8)

    # ---------- Stripe/background removal ----------
    # (A) Row-wise background (rolling mean) subtraction
    row_bg = cv2.blur(mean_u8, (1, ROW_BG_K))
    flat = cv2.normalize((mean_u8.astype(np.int16) - row_bg.astype(np.int16)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imwrite(os.path.join(OUTDIR, "02_flat_rowbg.png"), flat)

    # (B) Black-hat to emphasize dark blobs on bright-ish background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    bh = cv2.morphologyEx(mean_u8, cv2.MORPH_BLACKHAT, kernel)
    # cv2.imwrite(os.path.join(OUTDIR, "03_blackhat.png"), bh)

    # Choose the cleaner of (A) or (B) for thresholding:
    enface_clean = cv2.max(flat, bh)   # robust combo
    # cv2.imwrite(os.path.join(OUTDIR, "04_enface_clean.png"), enface_clean)

    # ---------- Threshold & morph ----------
    _, th = cv2.threshold(enface_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # blobs bright after black-hat/flatten

    th = cv2.medianBlur(th, 3)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
    cv2.imwrite(os.path.join(OUTDIR, "05_mask.png"), th)

    # ---------- Contours -> circular blobs ----------
    # Ensure blobs are white (255) foreground for contouring
    mask = th
    fg = mask
    # If blobs are black, invert:
    if np.mean(mask) > 127:      # white background, dark blobs → mean is high
        fg = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = th.shape
    # area_min = 0.0002*(H*W)
    # area_max = 0.02*(H*W)

    #  This is for laser calibration targets which are bigger
    area_min = 0.0005*(H*W)
    area_max = 0.1*(H*W)

    items = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < area_min or a > area_max:
            continue

        p = cv2.arcLength(cnt, True) + 1e-6
        circ = 4*np.pi*a/(p*p)

        # bounding box aspect ratio
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        aspect = w_ / float(h_)

        # collect, but store quality scores instead of hard-filtering too hard
        items.append(dict(
            cx = int(cv2.moments(cnt)["m10"] / (cv2.moments(cnt)["m00"]+1e-6)),
            cy = int(cv2.moments(cnt)["m01"] / (cv2.moments(cnt)["m00"]+1e-6)),
            r = cv2.minEnclosingCircle(cnt)[1],
            area = a,
            circ = circ,
            aspect = aspect,
            cnt = cnt
        ))

    # Now rank items by "roundness score" + area
    items = sorted(items, key=lambda d: (d["circ"], -d["area"]), reverse=True)

    # Keep only EXPECTED_N strongest blobs
    items = items[:EXPECTED_N]
    for i, it in enumerate(items):
        print(f"[{i}] area={it['area']:.1f}, circ={it['circ']:.2f}, aspect={it['aspect']:.2f}")


    centers = [(it["cx"], it["cy"]) for it in items]

    # Order 3x3 by rows then columns if we have 9
    def order_3x3(pts):
        if len(pts) != 9: return pts
        arr = np.array(pts)
        idx = np.argsort(arr[:,1])
        rows = np.array_split(idx, 3)
        rows = [r[np.argsort(arr[r,0])] for r in rows]
        return [tuple(arr[i]) for r in rows for i in r]
    centers_ord = order_3x3(centers)

    # ---------- Overlays & CSV ----------
    overlay = cv2.cvtColor(mean_u8, cv2.COLOR_GRAY2BGR)
    for it in items:
        cv2.drawContours(overlay, [it["cnt"]], -1, (0,255,0), 2)
    for k,(cx,cy) in enumerate(centers_ord, 1):
        cv2.circle(overlay, (cx,cy), 5, (0,0,255), -1)
        cv2.putText(overlay, f"{k}", (cx+6,cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUTDIR, "06_overlay_centers.png"), overlay)

    # fit Z-plane to all points
    coefs = opt.curve_fit(plane, np.hstack((x[:,None], y[:,None])).T, z)[0]
    normal = np.array([-coefs[1], -coefs[2], 1.0]) / np.linalg.norm([-coefs[1], -coefs[2], 1.0])
    print("Normal Vector: [{}, {}, {}]".format(normal[0], normal[1], normal[2]))
    print("Plane Parameters: z={:.3f}+{:.3f}x+{:.3f}y".format(*coefs))
    print("-------------------Make sure to copy the normal vector into the code! (laser_ablation.py)-------------------")

   # --- XY axes decided from (px ↔ mm) correlation; Z from plane normal ---
    H, W = overlay.shape[:2]
    origin = (40, H - 50)                   # corner anchor (px)
    L = max(30, min(H, W) // 12)            # arrow length (px)

    # Correlations to determine sign mapping
    def _sgn(v): return 1 if v >= 0 else -1
    corr_x = np.corrcoef(ix.astype(np.float32), x.astype(np.float32))[0, 1]
    corr_y = np.corrcoef(iy.astype(np.float32), y.astype(np.float32))[0, 1]
    # corr_z = np.corrcoef(z.astype(np.float32), iy.astype(np.float32))[0, 1]


    # +X in mm → image direction
    sx_img = _sgn(corr_x)                   # +1 = right, -1 = left
    # +Y in mm → image direction (iy increases downward)
    sy_img = 1 if corr_y >= 0 else -1       # +1 = down, -1 = up

    # Draw X (red)
    end_x = (origin[0] + sx_img * L, origin[1])
    cv2.arrowedLine(overlay, origin, end_x, (0, 0, 255), 2, tipLength=0.35)
    cv2.putText(overlay, "X", (end_x[0] + 6 * sx_img, end_x[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw Y (green)
    end_y = (origin[0], origin[1] + sy_img * L)
    cv2.arrowedLine(overlay, origin, end_y, (0, 255, 0), 2, tipLength=0.35)
    # place label outside arrow head
    y_lbl = (origin[0] - 15, origin[1] + sy_img * (L - 5))
    cv2.putText(overlay, "Y", y_lbl,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


    # # Z indicator (blue): ⊙ out of plane, ⊗ into plane
    # z_symbol = "o" if normal[2] >= 0 else "x"
    # cv2.putText(overlay, f"Z {z_symbol}", (origin[0] - 28, origin[1] - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(OUTDIR, "07_overlay_axes.png"), overlay)

    print("coefs:", coefs)



    with open(os.path.join(OUTDIR, "centers.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","cx_px","cy_px","est_x","est_y","est_z"])
        for i,(cx,cy) in enumerate(centers_ord,1):
            est_x = cx/sx + xmin; est_y = cy/sy + ymin
            est_z = plane([est_x, est_y], *coefs)
            w.writerow([i, cx, cy, est_x, est_y, est_z])

    print(f"Detected {len(centers_ord)} centers:", centers_ord)
    # print the xy values in mm
    for i,(cx,cy) in enumerate(centers_ord,1):
        est_x = cx/sx + xmin; est_y = cy/sy + ymin
        est_z = plane([est_x, est_y], *coefs)
        print(f"  {i}: pixel ({cx},{cy}) -> approx (x,y,z)=({est_x:.1f}mm, {est_y:.1f}mm, {plane([est_x, est_y], *coefs):.1f}mm)")
    print(f"[OK] Diagnostics in: {OUTDIR}")

def plane(x, A, B, C):
    return A + B*x[0] + C*x[1]



def detect_ring_centers_from_pcd_hough(
    pcd=None, npy_path=None, rgb_path=None,
    outdir="./ring_centers",
    grid=512,
    expected_n=9,                 # 9 => 3x3
    min_r=20, max_r=42,           # tune to your ring size (px)
    min_dist=55,                  # slightly below center spacing (px)
    dp=1.2, param1=120, param2=24,# Hough: dp, Canny hi, accumulator
    row_bg_k=201,                 # background flatten kernel
    invert=False                  # set True if rings are bright
):
    os.makedirs(outdir, exist_ok=True)

    # ---------- Load ----------
    if pcd is None:
        pts = np.load(npy_path); cols = np.load(rgb_path)
        print(f"Loaded {len(pts)} points from {npy_path} and {rgb_path}")
    else:
        p = o3d.io.read_point_cloud(pcd)
        pts = np.asarray(p.points); cols = np.asarray(p.colors)
        print(f"Loaded {len(pts)} points from {pcd}")

    x, y, z = pts[:,0], pts[:,1], pts[:,2]

    # ---------- Color -> gray (uint8) ----------
    if cols.ndim == 2 and cols.shape[1] == 3:
        c = cols.astype(np.float32);  c = (c*255.0) if c.max() <= 1.0 else c
        gray = (0.299*c[:,0] + 0.587*c[:,1] + 0.114*c[:,2]).astype(np.uint8)
    elif cols.ndim == 1:
        c = cols.astype(np.float32);  c = (c*255.0) if c.max() <= 1.0 else c
        gray = c.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected cols shape {cols.shape}")

    # ---------- XY → raster (same mapping as your function) ----------
    xmin, xmax = np.percentile(x, [0.5, 99.5]); xr = xmax - xmin
    ymin, ymax = np.percentile(y, [0.5, 99.5]); yr = ymax - ymin
    mx, my = 0.02*xr, 0.02*yr
    xmin -= mx; xmax += mx; ymin -= my; ymax += my
    sx = (grid-1)/(xmax-xmin + 1e-12)
    sy = (grid-1)/(ymax-ymin + 1e-12)
    ix = np.clip(((x - xmin)*sx).astype(np.int32), 0, grid-1)
    iy = np.clip(((y - ymin)*sy).astype(np.int32), 0, grid-1)

    sum_img = np.zeros((grid, grid), np.float32)
    cnt_img = np.zeros((grid, grid), np.int32)
    np.add.at(sum_img, (iy, ix), gray.astype(np.float32))
    np.add.at(cnt_img, (iy, ix), 1)
    mean_img = sum_img / np.maximum(cnt_img, 1)

    mean_u8 = cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    holes = ((cnt_img == 0).astype(np.uint8))*255
    mean_u8 = cv2.inpaint(mean_u8, holes, 3, cv2.INPAINT_TELEA)
    if invert: mean_u8 = cv2.bitwise_not(mean_u8)
    cv2.imwrite(os.path.join(outdir, "01_enface_mean.png"), mean_u8)

    # ---------- Ring enhancement (gentle) ----------
    row_bg = cv2.blur(mean_u8, (1, row_bg_k))
    flat = cv2.normalize((mean_u8.astype(np.int16) - row_bg.astype(np.int16)),
                         None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19,19))
    bh = cv2.morphologyEx(mean_u8, cv2.MORPH_BLACKHAT, kernel)
    work = cv2.max(flat, bh)
    work = cv2.GaussianBlur(work, (0,0), 1.2)

    # ---------- Hough circles ONLY ----------
    circles = cv2.HoughCircles(work, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_r, maxRadius=max_r)
    if circles is None:
        raise RuntimeError("No circles from Hough. Tune min_r/max_r/min_dist/param2.")

    circles = circles[0]  # (N,3) -> x,y,r

    # ---------- score circles: “ring darkness” along circumference ----------
    def ring_score(xc, yc, rc):
        th = np.linspace(0, 2*np.pi, 72, endpoint=False)
        xs = (xc + (rc-1)*np.cos(th)).astype(int)
        ys = (yc + (rc-1)*np.sin(th)).astype(int)
        m = (xs>=0)&(ys>=0)&(xs<work.shape[1])&(ys<work.shape[0])
        # Rings are dark: use response in `work` (higher = stronger edge) OR raw mean
        return float(work[ys[m], xs[m]].mean()) if m.any() else 0.0

    cand = [{"x": float(xc), "y": float(yc), "r": float(rc), "score": ring_score(xc,yc,rc)}
            for (xc, yc, rc) in circles]

    # ---------- Non-max suppression (keep one per neighborhood) ----------
    cand = sorted(cand, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in cand:
        if all((d["x"]-k["x"])**2 + (d["y"]-k["y"])**2 >= (min_dist*0.8)**2 for k in kept):
            kept.append(d)
        if len(kept) >= expected_n: break

    # ---------- Order 3×3 if 9 ----------
    if len(kept) == 9:
        arr = np.array([(d["x"], d["y"]) for d in kept])
        idx = np.argsort(arr[:,1])
        rows = np.array_split(idx, 3)
        rows = [r[np.argsort(arr[r,0])] for r in rows]
        kept = [kept[i] for r in rows for i in r]

    # ---------- Save overlay + CSV (px + mm) ----------
    overlay = cv2.cvtColor(mean_u8, cv2.COLOR_GRAY2BGR)
    with open(os.path.join(outdir, "ring_centers.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","cx_px","cy_px","r_px","est_x_mm","est_y_mm"])
        for i, d in enumerate(kept, 1):
            cx, cy, r = int(round(d["x"])), int(round(d["y"])), int(round(d["r"]))
            est_x = d["x"]/sx + xmin
            est_y = d["y"]/sy + ymin
            w.writerow([i, d["x"], d["y"], d["r"], est_x, est_y])
            cv2.circle(overlay, (cx,cy), r, (0,255,0), 2)
            cv2.circle(overlay, (cx,cy), 3, (0,0,255), -1)
            cv2.putText(overlay, str(i), (cx+6,cy-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(outdir, "ring_centers_overlay.png"), overlay)
    print(f"[OK] Hough kept {len(kept)} circles → {outdir}")
    centers_px = [(d["x"], d["y"]) for d in kept]
    centers_mm = [(d["x"]/sx + xmin, d["y"]/sy + ymin) for d in kept]
    return centers_px, centers_mm



def main():
    # 1. Load the point cloud
    # path = input("Enter path to PCD file: ")
    # path = "pointclouds/oct_raster_porcine.pcd"
    path = "/home/btl/Downloads/250911_powermeter/20250911-181726/oct_pointcloud.pcd"
    # path = "reliable_pointcloud_porcine.pcd"
    pcd = load_pcd(path, visualize=False, source="oct")  # realsense or oct
    
    # 2. Get selection mode
    # mode = select_region_mode()
    mode = 4
    
    # 3. Select region based on mode
    if mode == 1:
        # Point and radius mode
        selected_pcd = select_point_region(pcd)
        save_and_visualize(selected_pcd, "selected_region.pcd")
    
    elif mode == 2:
        # Rectangle region mode
        selected_pcd = select_rectangle_region(pcd)
        save_and_visualize(selected_pcd, "selected_region.pcd")
    
    elif mode == 3:
        # Square region with consistent waypoints
        seed_point = pick_point(pcd)
        print(f"Selected seed point: {seed_point}")
        
        
        # Ask for region size
        region_size = float(input("Enter region size in meters (default 0.03): ") or "0.03")
        
        # Select square region around seed point
        region_points, region_normals, region_pcd = select_square_region(
            pcd,
            seed_point,
            region_size=region_size,
            orient_consistent=True
        )        
        # Ask for cell size and height offset
        cell_size = float(input("Enter cell size in meters (default 0.01): ") or "0.01")
        height_offset = float(input("Enter height offset from average surface (default 0.03): ") or "0.03")
        
        # Generate grid waypoints with consistent orientation
        waypoints, stats_df, origin = generate_grid_waypoints(
            region_points, region_normals,
            cell_size=cell_size, height_offset=height_offset
                )

        visualize_waypoints(
            pcd,
            waypoints,
            normal_scale=cell_size,
            show_frames=False  # or False
        )


        # Save waypoints and statistics
        save_waypoints(waypoints)
        stats_df.to_csv("waypoints_stats.csv", index=False)
        save_and_visualize(region_pcd, "selected_region.pcd")
        
        print("Waypoint generation complete. Statistics saved to 'waypoints_stats.csv'.")

    elif mode == 4:
        # Laser center detection mode
        # npy_path = file_path + "npy_xyz.npy"
        # rgb_path = file_path + "npy_rgb.npy"
       
        find_laser_centers(pcd=path)

    elif mode == 5:
        # Example:
        centers_px, centers_mm = detect_ring_centers_from_pcd_hough(
            pcd=path,
            outdir="./laser_ring_centers",
            grid=512,
        )
         



if __name__ == "__main__":
    main()