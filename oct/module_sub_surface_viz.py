### Load the folder and generate point cloud for surface and one with subsurface
### for sub-surface,extract the pcd and add both. 


import numpy as np
import open3d
import copy
from module_oct_folder_viz import *
from subsurface_viz import *


path_of_folder = "/media/rp/Ubuntu_Data/OCT_Data/RATS/oct_250912_subsurface/250912_subsurface/20250912-215032/"

para_list_surface = {
        "data_type": "jpg", # jpg or nii
        "path_of_oct_data": path_of_folder,
        "flag_thres": "thres_bilateral_filter",
        "flag_vis": "false",
        "flag_label_mode": "unlabel",
        "flag_mode": "standard",  # calib_image 
        "kernel_gaussian": 3,
        "thres_low": 80,
        "thres_high": 255,
        "x_robot_stage": 0,
        "y_robot_stage": 0,
        "y_surf_cord": 30,    # only change this for surface and subsurface
        "idx_img_labelled_min": 88,
        "idx_img_labelled_max": 68,
        "pos_pixel_non_surface": 30,
        "num_b_scans": 256,
        "scale_x": 0.028/2,   # in mm
        "scale_y": 0.112/4,   # in mm
        "scale_z": 0.01459,   # in mm

    }
oct3d = Oct3DAnalysis()
para_list_subsurface = copy.deepcopy(para_list_surface)
para_list_subsurface["y_surf_cord"] = 70  # only change this


pcd_surface = oct3d.oct_unit_from_param(para_list_surface)
pcd_subsurface = oct3d.oct_unit_from_param(para_list_subsurface)

# extract only sub-surface part from pcd_subsurface
pcd_subsurface_extracted = extract_subsurface_pcd (path_of_pcd = path_of_folder + "oct_pointcloud.pcd", plane_dist=0.20, dbscan_eps=0.3, dbscan_min=15, save="subsurface.pcd", do_normals=False)
# check if denoising is needed through ball cluster



# display both pcd
open3d.visualization.draw_geometries([pcd_surface, pcd_subsurface_extracted], window_name="Surface and Sub-surface point clouds")





