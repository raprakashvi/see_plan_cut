import open3d as o3d

def visualize_point_cloud(pcd_file):
    """
    Visualizes a point cloud using Open3D.

    Args:
        pcd_file (str): Path to the point cloud file.
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Example usage
    pcd_file = "selected_region.pcd"
    visualize_point_cloud(pcd_file)