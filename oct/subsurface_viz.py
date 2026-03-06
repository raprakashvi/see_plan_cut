#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import copy
from pathlib import Path

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("NPY must be shape (N,3[+]).")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr[:, :3])
        return pcd
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load points or file empty: {path}")
    return pcd

def estimate_normals(pcd: o3d.geometry.PointCloud, k: int = 30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    pcd.normalize_normals()

def segment_top_plane(pcd: o3d.geometry.PointCloud, distance_threshold: float = 0.15,
                      ransac_n: int = 3, num_iters: int = 2000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iters)
    inliers = np.array(inliers, dtype=int)
    a, b, c, d = plane_model
    if c < 0:
        a, b, c, d = -a, -b, -c, -d
    plane_model = (a, b, c, d)
    return plane_model, inliers

def dbscan_clusters(pcd: o3d.geometry.PointCloud, eps: float = 0.3, min_points: int = 15):
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    return labels  # -1 = noise, otherwise 0..K-1

def color_by_labels(pcd: o3d.geometry.PointCloud, labels: np.ndarray, label_to_color=None):
    if label_to_color is None:
        uniq = np.unique(labels)
        colors = np.random.RandomState(42).rand(len(uniq), 3) * 0.8 + 0.2
        label_to_color = {lab: colors[i] for i, lab in enumerate(uniq)}
        if -1 in label_to_color:
            label_to_color[-1] = np.array([0.5, 0.5, 0.5])  # gray for noise
    # ensure all labels have a color
    uniq_labels = np.unique(labels)
    for lab in uniq_labels:
        if lab not in label_to_color:
            label_to_color[lab] = np.random.RandomState(lab + 100).rand(3) * 0.8 + 0.2
    cols = np.vstack([label_to_color[lab] for lab in labels])
    pc = copy.deepcopy(pcd)
    pc.colors = o3d.utility.Vector3dVector(cols)
    return pc

# --- Two draw helpers: NEW (Filament) vs LEGACY (GLFW) ---
def draw_modern(geoms, window_name="Open3D", width=1280, height=720):
    """Open3D Filament viewer (pretty)."""
    o3d.visualization.draw(geoms, title=window_name, width=width, height=height)

def draw_legacy(geoms, window_name="Open3D", width=1280, height=720):
    """Open3D legacy viewer (stable with VisualizerWithEditing)."""
    o3d.visualization.draw_geometries(geoms, window_name=window_name, width=width, height=height)

def pick_one_point_and_get_index(pcd: o3d.geometry.PointCloud) -> int:
    """
    Opens a legacy editor to pick points. Click one point, press 'Q' to close.
    Returns the picked point index (first pick) or raises if none picked.
    """
    # ### CHANGED: remove stray assignment to read_selection_polygon_volume
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick ONE point, then press 'Q'")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    indices = vis.get_picked_points()
    if not indices:
        raise RuntimeError("No point picked. Please run again and pick one point.")
    return indices[0]

def extract_subsurface_pcd (path_of_pcd , plane_dist=0.15, dbscan_eps=0.3, dbscan_min=15, save="subsurface.pcd", do_normals=False):
    # --- your fixed parameters (unchanged) ---
    in_path = path_of_pcd

    pcd = load_point_cloud(in_path)
    if do_normals:
        estimate_normals(pcd, k=30)

    plane_model, plane_inliers = segment_top_plane(pcd, distance_threshold=plane_dist)
    total_points = len(pcd.points)
    plane_mask = np.zeros(total_points, dtype=bool)
    plane_mask[plane_inliers] = True

    pcd_nonplane = pcd.select_by_index(plane_inliers, invert=True)
    if len(pcd_nonplane.points) == 0:
        raise RuntimeError("No non-plane points found; adjust plane_dist or check input.")

    cl_labels = dbscan_clusters(pcd_nonplane, eps=dbscan_eps, min_points=dbscan_min)
    cl = cl_labels.copy()
    good = cl_labels >= 0
    if np.any(good):
        uniq = np.unique(cl_labels[good])
        mapper = {lab: i+1 for i, lab in enumerate(uniq)}  # 1..K
        for k, v in mapper.items():
            cl[cl_labels == k] = v

    labels = np.full(total_points, -1, dtype=int)
    labels[plane_mask] = 0
    nonplane_indices = np.where(~plane_mask)[0]
    labels[nonplane_indices] = cl

    # --- preview: keep the NICE modern window (Filament) ---
    plane_color = np.array([0.9, 0.2, 0.2])
    colored = color_by_labels(pcd, labels, label_to_color={0: plane_color})
    print("Showing segments: red=top plane, other colors=clusters, gray=noise. Close window to continue.")
    draw_modern([colored], window_name="Segments: pick target next")  # ### still modern

    # --- pick with legacy ---
    print("Pick ONE point on the part you want to KEEP, then press 'Q'...")
    picked_idx = pick_one_point_and_get_index(colored)
    picked_label = labels[picked_idx]
    print(f"Picked point index = {picked_idx}, label = {picked_label}")

    # --- keep only that segment ---
    if picked_label == -1:
        kept_indices = np.array([picked_idx])
        print("Clicked on noise; keeping only the clicked point. (Tune dbscan_eps/dbscan_min.)")
    else:
        kept_indices = np.where(labels == picked_label)[0]

    kept = pcd.select_by_index(kept_indices.tolist(), invert=False)
    o3d.io.write_point_cloud(save, kept)
    print(f"Saved kept segment to: {save}  (points: {np.asarray(kept.points).shape[0]})")

    # --- show result with LEGACY viewer to avoid post-picking crash ---
    draw_legacy([kept.paint_uniform_color([0.1, 0.8, 0.3])],  # ### CHANGED
                window_name="Kept Segment")
    
    return kept

def main():
    # --- your fixed parameters (unchanged) ---
    in_path = "/media/rp/Ubuntu_Data/OCT_Data/RATS/oct_250912_subsurface/250912_subsurface/20250912-214737/oct_pointcloud.pcd"
    plane_dist = 0.15
    dbscan_eps = 0.3
    dbscan_min = 15
    save = "kept_segment.pcd"
    do_normals = False

    pcd = load_point_cloud(in_path)
    if do_normals:
        estimate_normals(pcd, k=30)

    plane_model, plane_inliers = segment_top_plane(pcd, distance_threshold=plane_dist)
    total_points = len(pcd.points)
    plane_mask = np.zeros(total_points, dtype=bool)
    plane_mask[plane_inliers] = True

    pcd_nonplane = pcd.select_by_index(plane_inliers, invert=True)
    if len(pcd_nonplane.points) == 0:
        raise RuntimeError("No non-plane points found; adjust plane_dist or check input.")

    cl_labels = dbscan_clusters(pcd_nonplane, eps=dbscan_eps, min_points=dbscan_min)
    cl = cl_labels.copy()
    good = cl_labels >= 0
    if np.any(good):
        uniq = np.unique(cl_labels[good])
        mapper = {lab: i+1 for i, lab in enumerate(uniq)}  # 1..K
        for k, v in mapper.items():
            cl[cl_labels == k] = v

    labels = np.full(total_points, -1, dtype=int)
    labels[plane_mask] = 0
    nonplane_indices = np.where(~plane_mask)[0]
    labels[nonplane_indices] = cl

    # --- preview: keep the NICE modern window (Filament) ---
    plane_color = np.array([0.9, 0.2, 0.2])
    colored = color_by_labels(pcd, labels, label_to_color={0: plane_color})
    print("Showing segments: red=top plane, other colors=clusters, gray=noise. Close window to continue.")
    draw_modern([colored], window_name="Segments: pick target next")  # ### still modern

    # --- pick with legacy ---
    print("Pick ONE point on the part you want to KEEP, then press 'Q'...")
    picked_idx = pick_one_point_and_get_index(colored)
    picked_label = labels[picked_idx]
    print(f"Picked point index = {picked_idx}, label = {picked_label}")

    # --- keep only that segment ---
    if picked_label == -1:
        kept_indices = np.array([picked_idx])
        print("Clicked on noise; keeping only the clicked point. (Tune dbscan_eps/dbscan_min.)")
    else:
        kept_indices = np.where(labels == picked_label)[0]

    kept = pcd.select_by_index(kept_indices.tolist(), invert=False)
    o3d.io.write_point_cloud(save, kept)
    print(f"Saved kept segment to: {save}  (points: {np.asarray(kept.points).shape[0]})")

    # --- show result with LEGACY viewer to avoid post-picking crash ---
    draw_legacy([kept.paint_uniform_color([0.1, 0.8, 0.3])],  # ### CHANGED
                window_name="Kept Segment")

if __name__ == "__main__":
    main()
