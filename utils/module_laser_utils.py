"""
This module contains functions to select points for resection, square for resection, and point define raster.
"""
import open3d as o3d
import numpy as np


class laser_utils:
    def __init__(self):
        a = 1

    def select_points_for_resection(self, pcd, laser_offset, oct_offset=0.105, num_points=1):
        """
        1. Load pointcloud from OCT (combined_pcd.pcd).
        2. Select a specified number of points in the point cloud.
        3. Return a list of selected points, each offset by the laser offset [0,0,0.005] m.
        """
        
        oct_pcd = o3d.io.read_point_cloud(pcd)
        print(f"Pick {num_points} points using [shift + left click].")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(oct_pcd)
        vis.run()
        vis.destroy_window()

        picked_points = vis.get_picked_points()
        if len(picked_points) < num_points:
            print(f"Insufficient points selected. You selected {len(picked_points)} out of {num_points} required.")
            return None

        selected_points = []
        for i, idx in enumerate(picked_points[:num_points]):
            point = np.asarray(oct_pcd.points)[idx]
            offset_point = point + laser_offset
            selected_points.append(offset_point)
            print(f"Selected point {i+1}: {offset_point}")

            # Display the selected point with a line joining the offset point to the original point
            line_points = [offset_point, point]
            lines = [[0, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(lines)))
            
            # add coordinates to the point cloud

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(oct_pcd)
            vis.add_geometry(line_set)
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0]))
            vis.run()
            vis.destroy_window()

        return selected_points

    

    def select_square_for_resection(self,pcd,laser_offset,oct_offset=0.105):
        """
        1. load pointcloud from OCT (combined_pcd.pcd)
        2. select 4 points in the point cloud to define a square
        3. return the points selected offset by the laser offset  [0,0,0.005] m 
        """
        
        oct_pcd = o3d.io.read_point_cloud(pcd)
        print("Pick 5 point using [shift + left click]")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(oct_pcd)
        vis.run()
        vis.destroy_window()

        picked_points = vis.get_picked_points()
        if len(picked_points) <= 4:
            print("No point selected. Exiting.")
            return None
        elif len(picked_points) > 5:
            print("Please select only 4 point. Exiting.")
            return None
        
        selected_points = np.asarray(oct_pcd.points)[picked_points]
        print(f"Selected points: {selected_points}")
        # offset the selected point by the laser offset
        selected_points += laser_offset    # only doing this for the z-axis but can be

        # display the selected points with a line joining the offset point to the original point
        line_points = [selected_points, selected_points - laser_offset]
        # Create lines to visualize offset
        line_points = np.vstack([selected_points, selected_points - laser_offset])
        lines = [[i, i + 5] for i in range(5)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
        
        # Visualize point cloud and lines
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(oct_pcd)
        vis.add_geometry(line_set)
        vis.run()
        vis.destroy_window()

        # can add the option to go call the OCT and scan the points

        return selected_points
    
    def select_point_define_raster(self, pcd, laser_offset, square_length=0.050, point_spacing=0.005, oct_offset=0.105):
        """
        1. load pointcloud from OCT (combined_pcd.pcd)
        2. select 1 point as raster center
        3. draw a square around the raster center 
        4. create points in a zig-zag raster pattern within the square
        5. visualize both original scan points and laser-offset points
        6. return the points selected offset by the laser offset [0,0,0.005] m
        """
        
        oct_pcd = o3d.io.read_point_cloud(pcd)
        print("Pick 1 point using [shift + left click]")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(oct_pcd)
        vis.run()
        vis.destroy_window()
        picked_points = vis.get_picked_points()
        
        if len(picked_points) == 0:
            print("No point selected. Exiting.")
            return None
        elif len(picked_points) > 1:
            print("Please select only one point. Exiting.")
            return None
        
        # Get the selected point as the center of the raster
        raster_center = np.asarray(oct_pcd.points)[picked_points[0]]
        
        # Define the square boundaries around the raster center
        half_square = square_length / 2
        print(f"Raster center: {raster_center}")
        print("Length of square: ", square_length)
        x_min, x_max = raster_center[0] - half_square, raster_center[0] + half_square
        y_min, y_max = raster_center[1] - half_square, raster_center[1] + half_square
        
        # Create zig-zag raster points
        num_points = int(square_length / point_spacing) + 1
        raster_points = []
        
        for i in range(num_points):
            y = y_min + i * point_spacing
            # Alternate direction for each row (zig-zag pattern)
            if i % 2 == 0:
                x_range = np.linspace(x_min, x_max, num_points)
            else:
                x_range = np.linspace(x_max, x_min, num_points)
                
            for x in x_range:
                point = np.array([x, y, raster_center[2]])
                raster_points.append(point)
        
        raster_points = np.array(raster_points)
        
        # Create offset points
        offset_points = raster_points + laser_offset
        
        # Visualize both original and offset points
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Create point cloud for raster points (red)
        raster_pcd = o3d.geometry.PointCloud()
        raster_pcd.points = o3d.utility.Vector3dVector(raster_points)
        raster_pcd.paint_uniform_color([1, 0, 0])  # Red
        
        # Create point cloud for offset points (blue)
        offset_pcd = o3d.geometry.PointCloud()
        offset_pcd.points = o3d.utility.Vector3dVector(offset_points)
        offset_pcd.paint_uniform_color([0, 0, 1])  # Blue
        
        # Add geometries to visualizer
        vis.add_geometry(oct_pcd)
        vis.add_geometry(raster_pcd)
        vis.add_geometry(offset_pcd)
        
        # Add lines to connect corresponding points
        for i in range(len(raster_points)):
            line = o3d.geometry.LineSet()
            points = o3d.utility.Vector3dVector([raster_points[i], offset_points[i]])
            lines = o3d.utility.Vector2iVector([[0, 1]])
            line.points = points
            line.lines = lines
            line.paint_uniform_color([0, 1, 0])  # Green lines
            vis.add_geometry(line)
        
        # Set view control for better visualization
        vis.get_render_option().point_size = 5.0
        vis.run()
        vis.destroy_window()
        
        return offset_points
            


        
        
    

if __name__ == "__main__":
    lu = laser_utils()
    pcd = # path to the pcd file
    laser_offset = np.array([0,0,0.005])
    # lu.select_points_for_resection(pcd,laser_offset,num_points=9)
    # lu.select_square_for_resection(pcd,laser_offset)

    # unit test for select_point_define_raster
    square_length = 0.010 # 10 mm
    point_spacing = 0.005 # 5 mm
    raster_points = lu.select_point_define_raster(pcd,laser_offset,square_length)