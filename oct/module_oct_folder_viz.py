 # %%
"""

 This module contains functions to visualize a folder of OCT images.
"""
import cv2
import numpy as np
import glob
import pandas as pd

import time
import copy
import matplotlib.pyplot as plt
import open3d
import nibabel as nib
# import module_oct_vol_scan as oct_vol
import os


class Oct3DAnalysis():

    def __init__(self):
        a = 1

    def oct_3d_array_from_path(self, para_list):
        """get the raw oct 3D image
        input: path of the oct raw data
            val_surface_level: (val_surface_level, length(img) - val_surface_level) -> find maximum values.
            pos_pixel_non_surface: the position values on the surface to be zeros.
        output: 3d array of the 3d image volume
        """

        # extract the parameters
        data_type = para_list.get("data_type", "jpg")  # default is jpg if not provided
        path_of_oct_data = para_list["path_of_oct_data"]
        flag_thres = para_list["flag_thres"]
        flag_mode = para_list["flag_mode"]
        kernel_gaussian = para_list["kernel_gaussian"]
        thres_low = para_list["thres_low"]
        thres_high = para_list["thres_high"]
        flag_vis = para_list["flag_vis"]
        y_surf_cord = para_list["y_surf_cord"]
        pos_pixel_non_surface = para_list["pos_pixel_non_surface"]
        flag_label_mode = para_list["flag_label_mode"]
        idx_img_labelled_min = para_list["idx_img_labelled_min"]
        idx_img_labelled_max = para_list["idx_img_labelled_max"]
        num_b_scans = para_list.get("num_b_scans", 128)  # default is 128 if not provided

        if flag_mode == "raw_image":
            input("raw oct 3d is a very big file for visualization, press enter to continue")
          
        # parameter setting for brightness pixels at each column (along the rows)
        bdry_width = 2
        len_of_fid = 30

        # print("path_of_oct_data = ", path_of_oct_data)

        if data_type == "jpg": 
            # get image folder information
            folderinfor = glob.glob(path_of_oct_data + "*.jpg")

            # print("folderinfor = ", folderinfor)
            num_of_img = len(folderinfor)
            # sort the folder information
            folderinfor = sorted(folderinfor)
            print(f"number of images: {num_of_img}")
            print("number of b scans = ", num_b_scans)

            # Adjust number of images to process based on num_b_scans
            step_size = max(1, num_of_img // num_b_scans)
            selected_images = folderinfor[::step_size][:num_b_scans]

        elif data_type == "nii":
            # if path_of_oct_data contains, .nii.gz file, then use the nii.gz file
            img1 = nib.load(path_of_oct_data)
            data = img1.get_fdata()
            affine = img1.affine
            print("data shape = ", data.shape)
            print("affine = ", affine)
            selected_images = data
            # add 1 dimension to the data
            selected_images = np.expand_dims(selected_images, axis=3)

        if data_type == "jpg":
            oct_img_array = np.zeros((len(selected_images), 512, 512, 3))
        elif data_type == "nii":
            oct_img_array = np.zeros((len(selected_images), 512, 512, 1))
            print("data type = nii")
        idx_img = 0
    
        """loop for each B-scan cross-sectional image"""
        # print("selected images = ", selected_images)
        for img_item in selected_images:
            if data_type!="nii":
                img_rgb = cv2.imread(img_item)
            else:
                img_rgb = img_item
            # print("img_rgb shape = ", img_rgb.shape)
            if img_rgb.shape[2] != 1:
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_rgb

            # enhance contrast using sigmoid function
            # img_rgb = self.sigmoid_contrast(img_rgb, cutoff=100, gain=1.5)


            if flag_mode == "segment_histeq": 
                """
                1. segment the surface image
                2. local histoequalization
                reference: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
                """
                img_copy = copy.deepcopy(img_gray)

                # global histogram equalization
                # img_histeq = cv2.equalizeHist(img_copy)

                # bilateral filter 
                para_bilateral_d = 8
                para_bilateral_sig_color = 300
                para_bilateral_sig_space = 150
                img_blur = cv2.bilateralFilter(img_gray, para_bilateral_d, para_bilateral_sig_color, para_bilateral_sig_space)
                # print("img_blur shape = ", img_blur.shape)
                # img_bdry = np.zeros((img_blur.shape[0], img_blur.shape[1]), np.uint8)

                # local histogram equalization
                kernel_size_localhisteq = 30
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(kernel_size_localhisteq, kernel_size_localhisteq))
                img_histlocaleq = clahe.apply(img_blur)
                img_output = cv2.cvtColor(img_histlocaleq, cv2.COLOR_GRAY2RGB) 

            if flag_mode == "standard": 
                """standard operation of the bscan images processing
                """

                # refer to the grayscale image
                img_copy = copy.deepcopy(img_gray)

                # bilateral filtering 
                para_bilateral_d = 8
                para_bilateral_sig_color = 300
                para_bilateral_sig_space = 150
                img_blur = cv2.bilateralFilter(img_gray.astype(np.uint8), para_bilateral_d, para_bilateral_sig_color, para_bilateral_sig_space)
                img_bdry = np.zeros((img_blur.shape[0], img_blur.shape[1]), np.uint8)
                img_high_contrast = img_blur

                """find the brighness column-pixel within the effective region"""
                xs = []  # x coordainte of the edge pixel
                ys = []  # y coordiante of the edge pixel
                for i in range(img_high_contrast.shape[1]):

                    # find the greatest pixel range from the (+30, bottom - 30)
                    y = np.argmax(img_high_contrast[y_surf_cord:-y_surf_cord, i]) + y_surf_cord
                    xs.append(i)
                    ys.append(y)

                    # find the average of the coordinates between the range
                    y_end = y + len_of_fid
                    y_avg = np.mean(img_high_contrast[y:y_end, i])

                    # set the width of the boundary
                    img_copy[y - bdry_width:y + bdry_width, i] = 255
                    img_bdry[y - bdry_width:y + bdry_width, i] = np.uint8(np.round(y_avg))

                x_surface_pixel = np.asarray(xs)
                y_surface_pixel = np.asarray(ys)
                if data_type == "nii":
                    img_output = img_bdry
                    # add the 3rd dimension
                    img_output = np.expand_dims(img_output, axis=2)
                    # print("img_output shape = ", img_output.shape)
                else:
                    img_output = cv2.cvtColor(img_bdry, cv2.COLOR_GRAY2RGB)


            if flag_mode == "calib_image":
                """this is a new algorithm
                1. get the surface coordinates 
                2. take the average intensity for each column the surface coordinates 
                3. replace the superficial pixel with the average coordinate
                """

                # case-1: in distance range 
                if (idx_img >= idx_img_labelled_min) and (idx_img <= idx_img_labelled_max):

                    if flag_label_mode == "unlabel": 

                        # label the image 
                        img_label_path = img_item
                        pts_edge_labelled = self.label_api.SingleImageLabel(img_input=img_label_path)
                        
                        # get labelled masked images 
                        img_label_ref = copy.deepcopy(img_rgb)
                        img_label_mask = self.Mask_from_interp_data(img_ref=img_label_ref, pts_edge_interp=pts_edge_labelled, flag_vis="false")
                        img_output = cv2.cvtColor(img_label_mask, cv2.COLOR_GRAY2RGB)
                        np.save(path_of_oct_data + str(idx_img) + ".npy", pts_edge_labelled)

                    elif flag_label_mode == "labelled": 
                        pts_edge = np.load(path_of_oct_data + str(idx_img) + ".npy") 
                        pts_edge_interp = self.interp_one_dimension(pts_edge, flag_vis="false")
                        img_label_ref = copy.deepcopy(img_rgb)
                        img_label_mask = self.Mask_from_interp_data(img_ref=img_label_ref, pts_edge_interp=pts_edge_interp, flag_vis="false")
                        img_output = cv2.cvtColor(img_label_mask, cv2.COLOR_GRAY2RGB)

                    else:
                        print("skip the labelling function")

                else:
                    img_copy = copy.deepcopy(img_gray)

                    # bilateral filtering 
                    para_bilateral_d = 8
                    para_bilateral_sig_color = 300
                    para_bilateral_sig_space = 150
                    img_blur = cv2.bilateralFilter(img_gray, para_bilateral_d, para_bilateral_sig_color, para_bilateral_sig_space)
                    img_bdry = np.zeros((img_blur.shape[0], img_blur.shape[1]), np.uint8)

                    # local histogram equalization
                    kernel_size_localhisteq = 30
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(kernel_size_localhisteq, kernel_size_localhisteq))
                    img_histlocaleq = clahe.apply(img_blur)

                    # get the contrast image 
                    img_high_contrast = img_histlocaleq
            
                    """find the brighness column-pixel within the effective region"""
                    xs = []  # x coordainte of the edge pixel
                    ys = []  # y coordiante of the edge pixel
                    for i in range(img_high_contrast.shape[1]):

                        # find the greatest pixel range from the (+30, bottom - 30)
                        y = np.argmax(img_high_contrast[y_surf_cord:-y_surf_cord, i]) + y_surf_cord
                        xs.append(i)
                        ys.append(y)

                        # find the average of the coordinates between the range
                        y_end = y + len_of_fid
                        y_avg = np.mean(img_high_contrast[y:y_end, i])

                        # set the width of the boundary
                        img_copy[y - bdry_width:y + bdry_width, i] = 255
                        img_bdry[y - bdry_width:y + bdry_width, i] = np.uint8(np.round(y_avg))

                    x_surface_pixel = np.asarray(xs)
                    y_surface_pixel = np.asarray(ys)
                    img_output = cv2.cvtColor(img_bdry, cv2.COLOR_GRAY2RGB)

            if flag_mode == "raw_image":
                img_output = img_rgb

            if flag_mode == "gray_filter":

                if flag_thres == "thres_gaussian_filter":

                    # filtering and smoothness
                    img_blur = self.oct_2d_class.blurring_filter(img_gray,
                                                                 flag_mode="gaussian",
                                                                 kernel_gaussian=kernel_gaussian)
                    img_output, _, img_thres = self.oct_2d_class.threshold_grayscale(img_gray=img_blur,
                                                                                     thres_low=thres_low,
                                                                                     thres_high=thres_high)

                    # remove the top layer pixel coordinates
                    img_thres[0 : pos_pixel_non_surface, 0 : 512] = 0

                    img_gray_cluster = cv2.bitwise_and(img_blur, img_thres)
                    img_output = cv2.cvtColor(img_gray_cluster, cv2.COLOR_GRAY2RGB)

                elif flag_thres == "thres_bilateral_filter":

                    # Code to find phantom surface boundary

                    # create a image copy and bilateral filter : preserves boundary
                    img_copy = copy.deepcopy(img_gray)

                    #(src, diameter of each pixel neighborhood,sigmacolor,sigmaspace)
                    para_bilat_d = 8
                    para_bilat_sig_color = 300
                    para_bilat_sig_space = 150
                    img_blur = cv2.bilateralFilter(img_gray, para_bilat_d, para_bilat_sig_color, para_bilat_sig_space)

                    # create a black canvas for boundaries
                    img_bdry = np.zeros((img_blur.shape[0],img_blur.shape[1]), np.uint8)

                    """find the brighness column-pixel within the effective region"""
                    xs = []     # x coordainte of the edge pixel
                    ys = []     # y coordiante of the edge pixel
                    y_surf_cord = 30        # range in (top 30 pixels and bottom-up 30 pixels)
                    bdry_width = 2
                    for i in range(img_blur.shape[1]):
                        # find the greatest pixel range from the (+30, bottom - 30)
                        y = np.argmax(img_blur[y_surf_cord:-y_surf_cord, i]) + y_surf_cord
                        xs.append(i)
                        ys.append(y)
                        img_copy[y-bdry_width:y+bdry_width, i] = 255 # the width of boundary is four rows
                        img_bdry[y-bdry_width:y+bdry_width, i] = 255
                    x_surface_pixel = np.asarray(xs)
                    y_surface_pixel = np.asarray(ys)
                    # apply the threshold image
                    img_bdry = cv2.bitwise_and(img_copy, img_bdry)

                    # scale to the rgb value
                    img_output = cv2.cvtColor(img_bdry, cv2.COLOR_GRAY2RGB)

            # Create overlay with detected surface in light red
            if 'x_surface_pixel' in locals() and 'y_surface_pixel' in locals():
                # Create a copy of the original image for overlay
                img_overlay = copy.deepcopy(img_rgb)
                
                # Draw surface points in light red (BGR format: light red = [128, 128, 255])
                for x, y in zip(x_surface_pixel, y_surface_pixel):
                    cv2.circle(img_overlay, (int(x), int(y)), 1, (128, 128, 255), -1)
                
                # Blend the overlay with the original image (0.7 background, 0.3 overlay)
                img_display = cv2.addWeighted(img_rgb, 0.7, img_overlay, 0.3, 0)
            else:
                img_display = img_output

            # display the image
            cv2.imshow("img_output", img_display)
            cv2.waitKey(10)
            time.sleep(0.1)

            # put the 2D image to the 3d-oct array 
            oct_img_array[idx_img, :, :] = img_output

            # visualize the programs
            if flag_vis == "true":
                cv2.imshow("img_output", img_output)
                # cv2.imshow("img_copy", img_copy)
                cv2.waitKey(10)
                time.sleep(0.1)
                # # checking mode
                plt.imshow(img_output)
                plt.scatter(x_surface_pixel, y_surface_pixel, c= "red", marker='.', linestyle=':')
                plt.plot(x_surface_pixel, y_surface_pixel)
                plt.show()

            # update the index
            idx_img = idx_img + 1

        cv2.destroyAllWindows()

        if data_type == "nii":
            oct_img_array = np.squeeze(oct_img_array, axis=3)
            return oct_img_array, affine

        return oct_img_array
    
    def sigmoid_contrast(self, gray, cutoff=110.0, gain=0.035):
        # Work in float then back to uint8
        g = gray.astype(np.float32)
        # Prevent overflow by clipping the exponential argument
        exp_arg = -gain * (g - cutoff)
        exp_arg = np.clip(exp_arg, -500, 500)  # Prevent overflow/underflow
        s = 1.0 / (1.0 + np.exp(exp_arg))
        s = (s * 255.0).clip(0, 255).astype(np.uint8)
        return s

    def pixel_to_3d_pos(self, data_img_array, num_b_scans=128, scale_img=1.0, scale_x=0.0247, scale_y=0.1, scale_z=0.01459):
        """Modifies the data from pixel scale to mm scale by using experimentally determined scaling factors
        Right hand axis rule for determining X Y Z axis
        argvs:
            scale_img = 1/4, downsampling the image
            scale_x = 0.05688 mm, width per pixel of full width OCT scan
            scale_y = 0.01 mm, for total length = 12.8mm, and 128 scans , scale_y = 12.8/128
            scale_z = 0.01459 mm, depth per pixel of full width OCT scan
        returns:
            output:
        """
        # Scaling adjustments based on the number of B-scans
        # scale_factor = 128 / num_b_scans
        # scale_factor = 256 / num_b_scans

        # # scale_y *= scale_factor
        print("scale factors = ", scale_x, scale_y, scale_z)
        print("num of b-scans = ", num_b_scans)

        # initialize a data structure
        df_oct = pd.DataFrame()

        for i in range(data_img_array.shape[0]):

            # print("oct 3d index = ", i)

            # load the current image
            img_rgb = data_img_array[i, :, :].astype(np.uint8)



            # resize + grayscale
            # TODO: check -> not allow to scale the image then back-scale the image again.
            # assert int(scale_img) == 1
            img_rgb = cv2.resize(img_rgb, None, fx=scale_img, fy=scale_img)
            img_gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            # exract r,g,b channel 2d matricies
            r = img_rgb[:, :, 0]
            g = img_rgb[:, :, 1]
            b = img_rgb[:, :, 2]

            # coordinate transform (pixel to unit)
            idx_final = list(range(0, img_gray.shape[0] * img_gray.shape[1]))
            [row, col] = np.unravel_index(idx_final, img_gray.shape, 'F')

            # 3d data array
            x_temp = row / scale_img * scale_x
            z_temp = -1 * col / scale_img * scale_z
            y_temp = -np.ones(x_temp.shape) * i * scale_y
            idx_use = np.where( img_gray.flatten() != 0 )[0]
            pcd_data = [x_temp[idx_use], y_temp[idx_use], z_temp[idx_use],
                        r.flatten()[idx_use], g.flatten()[idx_use], b.flatten()[idx_use],
                        img_gray.flatten()[idx_use]]
            npy_data_from_pdf = np.asarray(pcd_data).transpose()
            df_temp = pd.DataFrame( npy_data_from_pdf )
            df_oct = pd.concat([df_oct, df_temp])
        return df_oct

    def dfConvert(self, df_oct, flag_check="masked"):
        """Extracts the XYZ and RGB and Intensity value from the dataframe of pointcloud
        """

        # read all rows in columns 0,1,2 (xyz)
        df_xyz = df_oct.iloc[:, 0:3]
        # read all rows in column 3,4,5 (r,g,b)
        df_rgb = df_oct.iloc[:, 3:6]
        # read all rows in column 6 (intensity)
        df_intensity = df_oct.iloc[:, 6]

        if flag_check == "masked":
            # mask out all parts of the dataframe that have 0 intensity
            mask = df_intensity != 0
            # apply mask to the position and label(rgb) dataframes
            xyz = df_xyz[mask]
            rgb = df_rgb[mask]

            return xyz, mask, rgb

        return df_xyz, df_intensity, df_rgb

    def oct_unit(self, path_of_oct_data, num_b_scans=128,scale_x=0.028/2, scale_y=0.112/4, scale_z=0.01459):
        # single unit test with the example data
        para_list = {
            "data_type": "jpg", # jpg or nii
            "path_of_oct_data": path_of_oct_data,
            "flag_thres": "thres_bilateral_filter",
            "flag_vis": "false",
            "flag_label_mode": "unlabel",
            "flag_mode": "standard",  # calib_image 
            "kernel_gaussian": 3,
            "thres_low": 80,
            "thres_high": 255,
            "x_robot_stage": 0,
            "y_robot_stage": 0,
            "y_surf_cord": 30,
            "idx_img_labelled_min": 88,
            "idx_img_labelled_max": 68,
            "pos_pixel_non_surface": 30,
            "num_b_scans": num_b_scans
        }

        path_data_save = para_list["path_of_oct_data"]
        if para_list["data_type"] == "nii":
            oct_img_array , affine = self.oct_3d_array_from_path(para_list)
            # save the nii file
            modified_nifti_img = nib.Nifti1Image(oct_img_array, affine)
            nib.save(modified_nifti_img, path_of_oct_data.replace(".nii.gz", "_modified.nii.gz"))
            exit()

        else:
            oct_img_array = self.oct_3d_array_from_path(para_list)
        print("oct_img_array shape = ", oct_img_array.shape)
        

        df_oct = self.pixel_to_3d_pos(data_img_array=oct_img_array, num_b_scans=num_b_scans, scale_img=1.0,
                                        scale_x=scale_x, scale_y=scale_y, scale_z=scale_z)
        print("df_oct shape = ", df_oct.shape)
        print("df_oct = ", df_oct.head())
        df_xyz, df_intensity, df_rgb = self.dfConvert(df_oct, flag_check="masked")
        print("df_xyz shape = ", df_xyz.shape)
        print("df_xyz head = ", df_xyz.head())  

        # open3D pointcloud
        pcd = open3d.geometry.PointCloud()
        pts_xyz = df_xyz.to_numpy()
        pts_rgb = df_rgb.to_numpy()
        pcd.colors = open3d.utility.Vector3dVector(pts_rgb/255)
        pcd.points = open3d.utility.Vector3dVector(pts_xyz)
        #open3d.visualization.draw_geometries([pcd]) # # Optional: estimate normals for better rendering
        # pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
        #     radius=0.1, max_nn=30))
        # pcd.orient_normals_consistent_tangent_plane(100)

        #open3d.visualization.draw_geometries([pcd])
        np.save(path_data_save + "npy_xyz.npy", np.asarray(df_xyz))
        np.save(path_data_save + "npy_rgb.npy", np.asarray(df_rgb))

        # also save as pcd file
        open3d.io.write_point_cloud(path_data_save + "oct_pointcloud.pcd", pcd)


        print("Range of X: ", np.min(df_xyz[0]), np.max(df_xyz[0]))
        print("Range of Y: ", np.min(df_xyz[1]), np.max(df_xyz[1]))
        print("Range of Z: ", np.min(df_xyz[2]), np.max(df_xyz[2]))

   

    def oct_unit_from_param(self,para_list):

        path_data_save = para_list["path_of_oct_data"]
        path_of_oct_data = para_list["path_of_oct_data"]
        num_b_scans = para_list.get("num_b_scans", 256)
        scale_x = para_list.get("scale_x", 0.028/2)
        scale_y = para_list.get("scale_y", 0.112/4)
        scale_z = para_list.get("scale_z", 0.01459)


        if para_list["data_type"] == "nii":
            oct_img_array , affine = self.oct_3d_array_from_path(para_list)
            # save the nii file
            modified_nifti_img = nib.Nifti1Image(oct_img_array, affine)
            nib.save(modified_nifti_img, path_of_oct_data.replace(".nii.gz", "_modified.nii.gz"))
            exit()

        else:
            oct_img_array = self.oct_3d_array_from_path(para_list)
        print("oct_img_array shape = ", oct_img_array.shape)
        

        df_oct = self.pixel_to_3d_pos(data_img_array=oct_img_array, num_b_scans=num_b_scans, scale_img=1.0,
                                        scale_x=scale_x, scale_y=scale_y, scale_z=scale_z)
        print("df_oct shape = ", df_oct.shape)
        print("df_oct = ", df_oct.head())
        df_xyz, df_intensity, df_rgb = self.dfConvert(df_oct, flag_check="masked")
        print("df_xyz shape = ", df_xyz.shape)
        print("df_xyz head = ", df_xyz.head())  

        # open3D pointcloud
        pcd = open3d.geometry.PointCloud()
        pts_xyz = df_xyz.to_numpy()
        pts_rgb = df_rgb.to_numpy()
        pcd.colors = open3d.utility.Vector3dVector(pts_rgb/255)
        pcd.points = open3d.utility.Vector3dVector(pts_xyz)
        #open3d.visualization.draw_geometries([pcd]) # # Optional: estimate normals for better rendering
        # pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
        #     radius=0.1, max_nn=30))
        # pcd.orient_normals_consistent_tangent_plane(100)

        #open3d.visualization.draw_geometries([pcd])
        np.save(path_data_save + "npy_xyz.npy", np.asarray(df_xyz))
        np.save(path_data_save + "npy_rgb.npy", np.asarray(df_rgb))

        # also save as pcd file
        open3d.io.write_point_cloud(path_data_save + "oct_pointcloud.pcd", pcd)


        print("Range of X: ", np.min(df_xyz[0]), np.max(df_xyz[0]))
        print("Range of Y: ", np.min(df_xyz[1]), np.max(df_xyz[1]))
        print("Range of Z: ", np.min(df_xyz[2]), np.max(df_xyz[2]))

        return pcd


    def convert_npy_pcd (self,path_data_save,save_location, file_idx = "/215_53"):
        # convert the npy file to the pcd file
        pts_xyz = np.load(path_data_save + file_idx + "_npy_xyz.npy")
        pts_rgb = np.load(path_data_save + file_idx + "_npy_rgb.npy")

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts_xyz)
        # pcd.colors = open3d.utility.Vector3dVector(pts_rgb/255)
        #open3d.visualization.draw_geometries([pcd])
        open3d.io.write_point_cloud(save_location + "/test.pcd", pcd)

       


if __name__ == "__main__":
    scale_x = 0.028 / 2  # mm # half width , 128 scans, total width = 14mm (number of b scans agnostic)
    scale_y = 0.112 / 4  # mm  # half width , 256 B scans : scale_y = scale_x * 512 / num_b_scans
    scale_z = 0.01459  # mm # depth per pixel of full width OCT scan: stays constant
    oct_3d_class = Oct3DAnalysis()
    # path_main_folder = "/home/btl/Desktop/dataset/rp_vw_fun/full_raster_test_2mm_11-5_2nd/"
    path_numpy_save = r"E:\hi080\Documents\Research\Laser MPC\RATS_Dataset\LTI_Tests\250911_LTI_testing\250911_LTI_90\\"
    # # single folder

    path_main_folder = r"E:\hi080\Documents\Research\Laser MPC\RATS_Dataset\LTI_Tests\250911_LTI_testing\250911_LTI_90\\"

    folder_list = [f for f in os.listdir(path_main_folder) if os.path.isdir(os.path.join(path_main_folder, f))]   
    for i in range(len(folder_list)):
        file_name = os.path.join(path_main_folder, folder_list[i]) + "/"
        print("file_name = ", file_name)
        oct_3d_class.oct_unit(path_of_oct_data=file_name, num_b_scans=256, scale_x=scale_x, scale_y=scale_y, scale_z=scale_z)
