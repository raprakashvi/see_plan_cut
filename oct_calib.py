# Useful Aids:
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html (General calibration tutorial)
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b (hand-eye calibration documentation)

import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2 as cv
import rtde_control
import rtde_receive
import scipy 
import os
from oct.module_oct_vol_scan import oct_raster_scan
import pyvista as pv

def ur2rotmat(tcpPose):
    """
    Convert UR TCP pose to rotation matrix.
    tcpPose - 6D pose of the TCP in the form [x, y, z, rx, ry, rz]
    """
    rvec = np.array(tcpPose[3:])
    angle = np.linalg.norm(rvec)
    unitrvec = rvec / angle
    translation, rotation = tcpPose[0:3], (unitrvec[0] * np.sin(angle/2), unitrvec[1] * np.sin(angle/2), unitrvec[2] * np.sin(angle/2), np.cos(angle/2))  
    
    # Quaternion to rot. matrix: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    rotMat = np.array([[2 * (rotation[3]**2+rotation[0]**2) - 1, 2 * (rotation[0]*rotation[1]-rotation[3]*rotation[2]), 2 * (rotation[0]*rotation[2]+rotation[3]*rotation[1]), translation[0]], 
                [2 * (rotation[0]*rotation[1]+rotation[3]*rotation[2]), 2 * (rotation[3]**2+rotation[1]**2) - 1, 2 * (rotation[1]*rotation[2]-rotation[3]*rotation[0]), translation[1]],
                [2 * (rotation[0]*rotation[2]-rotation[3]*rotation[1]), 2 * (rotation[1]*rotation[2]+rotation[3]*rotation[0]), 2 * (rotation[3]**2+rotation[2]**2) - 1, translation[2]],
                [0, 0, 0, 1]])
    return rotMat

def runOCTScan(poses, ip):
    '''
    Function to scan at a list of poses.

    Parameters
    ----------
    poses : Nx6 Array
        List containing joint values in radians.
    ip : String
        Robot IP Address.

    Returns
    -------
    Nx4x4 Array
        List of homogenous transform matrices to the robot end effector at each scan pose.

    '''
    # Connect to robot
    ctrlInterface = rtde_control.RTDEControlInterface(ip)
    rcvInterface = rtde_receive.RTDEReceiveInterface(ip)
    
    # Initialize FK List and OCT System
    fkMats = []
    oct_scan = oct_raster_scan()
    
    # Go through poses and run OCT scan
    for i, pose in enumerate(poses):
        input("Robot will move to pose {}. Press Enter to continue...".format(pose))
        ctrlInterface.moveJ(pose, 0.1, 2)
        time.sleep(3)
        scan_flag = oct_scan.start_oct_scan(f"{i}")
        if not scan_flag:
            print("OCT scan failed. Exiting...")
            break
        elif scan_flag:
            print("OCT scan successful.")
            
        # Save all FK Matrices
        tcpPose = rcvInterface.getActualTCPPose() # Get robot FK matrix
        fkMats.append(ur2rotmat(tcpPose))
        
    # Disconnect and cleanup
    ctrlInterface.stopScript()
    ctrlInterface.disconnect()
    rcvInterface.disconnect()
    return np.array(fkMats)

def findCircleCoords(points, colors, abRatio = 4, param2 = 15, radiusLim = [1, 100], ydot = False):
    '''
    Function to detect circle centers c_o and c_x and c_y given a uniform gridded point cloud (e.g. from OCT scans).

    Parameters
    ----------
    points : Nx3 Array
        Array of xyz points. Should be in a structured grid format.
    colors : Nx3 Array
        Array of RGB values ranging from 0-255.
    abRatio : float, optional
        The ratio of B-scan distance to A-scan distance. The default assumes a square aspect ratio, 512 A-scans per B-scan, and 128 B-scans per C-scan, giving a ratio of 512/128 = 4.
    param2 : int, optional
        param2 passed into cv2.HoughCircles. Lower values detect more circles, but loses accuracy. The default is 15.
    radiusLim : (int, int), optional
        The minimum and maximum radius to detect. The default is (1, 100).
    ydot : Boolean, optional
        Whether the calibration pattern has a ydot (versus only an origin and xdot). Defaults to false.

    Returns
    -------
    c_o : 1x3 Array
        xyz Coordinate of the center of the "origin" dot in the marker frame.
    c_x : 1x3 Array
        xyz Coordinate of the center of the "x-axis" dot in the marker frame.
    c_y : 1x3 Array (Optional Return)
        xyz Coordinate of the center of the "y-axis" dot in the marker frame. If ydot is false, this evaluates to None.

    '''
    
    c_o = None
    c_x = None
    c_y = None
    
    # Get enface projection as image array
    xVals = np.unique(points[:,0])
    yVals = np.unique(points[:,1])
    colorArray = np.zeros((len(xVals), len(yVals)))
    for i, x in enumerate(xVals):
        for j, y in enumerate(yVals):
            colorInds = np.where(np.logical_and(points[:,0] == x, points[:,1] == y))
            if len(colorInds[0]) == 0:
                print("Error detected when converting to 2D Image. Ensure the data is in a gridded format.")
                return None
            # Take mean of all points at a given X/Y index and cast down to greyscale from RGB
            colorArray[i, j] = np.mean(colors[colorInds[0]])
            
    # Perform circle detection, based on https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
    gray = cv.medianBlur(colorArray.astype('uint8'), 5)
    gray = cv.resize(gray, (gray.shape[0], gray.shape[1] * abRatio)) # Interpolate to ensure square aspect ratio pixels
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=50, param2=param2,
                                   minRadius=radiusLim[0], maxRadius=radiusLim[1])[0]
    
    # Label circles and convert back to xyz coordinates from pixel coordinates
    if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles:
                drawgray = gray.copy()
                center = (i[0], i[1])
                # circle center
                cv.circle(drawgray, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(drawgray, center, radius, (255, 0, 255), 3)
                cv.imshow("detected circles", drawgray)
                cv.waitKey(0)
                
                # Take User input to label as c_o or c_x
                if ydot:
                    selection = int(input("Label the selected circle:\n1: Origin\n2: X-dot\n3: Y-dot\n4: None\n: "))
                    if selection == 1:
                        center = i.copy()
                        center[0] /= abRatio
                        xyCenter = (xVals[center[1]], yVals[center[0]])
                        c_o = np.array([[xyCenter[0], xyCenter[1], np.mean(points[np.where(np.logical_and(points[:,0] == xyCenter[0], points[:,1] == xyCenter[1]))[0], 2])]])
                    elif selection == 2:
                        center = i.copy()
                        center[0] /= abRatio
                        xyCenter = (xVals[center[1]], yVals[center[0]])
                        c_x = np.array([[xyCenter[0], xyCenter[1], np.mean(points[np.where(np.logical_and(points[:,0] == xyCenter[0], points[:,1] == xyCenter[1]))[0], 2])]])
                    elif selection == 3:
                        continue
                else:
                    selection = int(input("Label the selected circle:\n1: Origin\n2: X-dot\n3: None\n: "))
                    if selection == 1:
                        center = i.copy()
                        center[0] /= abRatio
                        xyCenter = (xVals[center[1]], yVals[center[0]])
                        c_o = np.array([[xyCenter[0], xyCenter[1], np.mean(points[np.where(np.logical_and(points[:,0] == xyCenter[0], points[:,1] == xyCenter[1]))[0], 2])]])
                    elif selection == 2:
                        center = i.copy()
                        center[0] /= abRatio
                        xyCenter = (xVals[center[1]], yVals[center[0]])
                        c_x = np.array([[xyCenter[0], xyCenter[1], np.mean(points[np.where(np.logical_and(points[:,0] == xyCenter[0], points[:,1] == xyCenter[1]))[0], 2])]])
                    elif selection == 3:
                        center = i.copy()
                        center[0] /= abRatio
                        xyCenter = (xVals[center[1]], yVals[center[0]])
                        c_y = np.array([[xyCenter[0], xyCenter[1], np.mean(points[np.where(np.logical_and(points[:,0] == xyCenter[0], points[:,1] == xyCenter[1]))[0], 2])]])
                    elif selection == 4:
                        continue
    else:
        print("No circles detected!")
        return None

    # Return values only if both c_o and c_x were properly detected and labeled
    if ydot and c_o is not None and c_x is not None and c_y is not None:
        return c_o, c_x, c_y
    elif not ydot and c_o is not None and c_x is not None:
        return c_o, c_x, c_y
    else:
        print("Not enough circles were detected!")
        return None

if __name__ == "__main__":
    # PARAMETERS TO EDIT --------------------------------------------
    # List of poses to use for calibration
    poses = np.array([[97.28, -31.27, 91.09, -250.81, -70.47, 97.12],
                      [93.51, -31.42, 87.52, -242.91, -47.74, 97.13],
                      [90.63, -26.20, 72.22, -243.88, -7.55, 99.10],
                      [92.11, -30.77, 82.29, -229.90, -37.25, 99.10],
                      [92.51, -31.77, 82.16, -218.99, -42.38, 99.08],
                      [97.15, -34.23, 90.04, -231.22, -52.04, 79.72]]) / 180 * np.pi # Add poses here
    
    ip = "192.168.1.103"
    markerScaleFactor = 0.00202 # distance between individual markers within group of 2
    # -------------------------------------------------------------
    
    reprojectionIndices = range(len(poses))
    step = int(input("Steps:\n  1. Collect scans at poses\n  2. Process OCT scans into surfaces (done in separate script)\n  3. Calculate calibration\nEnter Step Number (1, 2, 3): "))
    
    # STEP 1: Get scans
    if step == 1:
        fkMats = runOCTScan(poses, ip)
        np.save("./data/calibrationOCT/fkMat.npy", fkMats)
        print("FK Data has been saved to data/calibrationOCT/fkMat.npy. OCT Files should be taken from the OCT computer, analyzed, then placed in data/calibrationOCT/scans with filename npy_xyz.npy as a Nx3 array with color data as npy_rgb.npy.")
    
    # STEP 2: Process OCT scans into surfaces (done in separate script, files must be transfered from OCT computer)
    elif step == 2:
        print("Step 2 is done with a separate script!")
        
    # STEP 3:
    # Note that these names are flipped from https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b because our convention/notation is different
    elif step == 3:
        input("Make sure that the OCT scan surfaces are saved in individual folders in data/calibrationOCT/scans with filename npy_xyz.npy as a Nx3 array with color data as npy_rgb.npy! Press Enter to confirm.")
        
        # Determines whether a 2-dot or 3-dot calibration method will be used.
            # 2-dot: X-axis = Vector between two dots, Z-axis = Vector normal to planar fit, Y-axis = np.cross(Z-axis, X-axis)
            # 3-dot: X-axis = Vector between x-dot and origin, Y-axis = Vector between y-dot and origin, Z-axis = np.cross(X-axis, Y-axis)
        method = int(input("Select calibration method.\n 1: 2-marker\n 2: 3-marker\n:"))
        R_base2ee = [] # rotation matrix from base to end effector
        t_base2ee = [] # translation vector from base to end effector
        R_cam2obj = [] # rotation matrix from camera to object
        t_cam2obj = [] # translation vector from camera to object
        fkMats = np.load("./data/calibrationOCT/fkMat.npy")
        
        if method == 1:
            circleCenters = np.zeros((len(fkMats), 2, 3)) # Axis one is indexed as 0 = Origin, 1 = X-axis, 2 = Y-axis
        elif method == 2:
            circleCenters = np.zeros((len(fkMats), 3, 3)) # Axis one is indexed as 0 = Origin, 1 = X-axis
        circleCenters[:] = np.nan
        
        path = r"data/calibrationOCT/scans"
        folders = os.listdir(path)
        
        # Main loop to go through all collected scans
        for i in range(fkMats.shape[0]):
            
            print("Pose {}...".format(i+1))
            
            # Load data (points and colors both Nx3 array) 
            points = np.load(os.path.join(path, folders[i], "npy_xyz.npy"))
            colors = np.load(os.path.join(path, folders[i], "npy_rgb.npy"))
            
            # find circle centers from enface projection c_o, c_x (length-3 vectors)
            try:
                c_o, c_x, c_y = findCircleCoords(points, colors, abRatio = 2, ydot = (method == 2))
                c_o = c_o / 1000 # mm to m
                circleCenters[i][0] = c_o.copy()
                c_x = c_x / 1000 # mm to m
                circleCenters[i][1] = c_x.copy()
                if method == 2:
                    c_y = c_y / 1000 # mm to m
                    circleCenters[i][2] = c_y.copy()
                
            except TypeError as e:
                continue
            
            # get the absolute center and vector directions for XYZ axes along the marker axis in OCT frame
            if method == 1:
                xvec = (c_x - c_o) / np.linalg.norm((c_x - c_o))
                A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
                C,_,_,_ = scipy.linalg.lstsq(A, points[:,2])    # coefficients
                zvec = np.array([-C[0], -C[1], 1]) / np.linalg.norm(np.array([-C[0], -C[1], 1]))
                yvec = np.cross(zvec, xvec) / np.linalg.norm(np.cross(zvec, xvec))
                zvec = zvec[None,:]
            elif method == 2:
                xvec = (c_x - c_o) / np.linalg.norm((c_x - c_o)) 
                yvec = (c_o - c_y) / np.linalg.norm((c_o - c_y)) # Flip the Y-vector due to how the calibration grid is oriented, to ensure Z points upward (not critical, but used for consistency with oct_calib_2marker.py)
                zvec = np.cross(xvec, yvec) / np.linalg.norm(np.cross(xvec, yvec))         
                
            # find the absolute marker frame origin, which is at the c_o center
            origin = c_o
            
            # get transform bet. frames
            rmat = np.hstack((xvec.T, yvec.T, zvec.T))
            tvec = origin.copy()
            
            # Plot data
            pl = pv.Plotter()
            pl.add_mesh(points, scalars=colors)
            pl.add_mesh(pv.Arrow(start = (c_o * 1000), direction = xvec, scale='auto'), color='red')
            pl.add_mesh(pv.Arrow(start = (c_o * 1000), direction = yvec, scale='auto'), color='green')
            pl.add_mesh(pv.Arrow(start = (c_o * 1000), direction = zvec, scale='auto'), color='blue')
            if method == 1:
                pl.add_mesh(np.vstack((c_o * 1000, c_x * 1000)), point_size = 40)
            elif method == 2:
                pl.add_mesh(np.vstack((c_o * 1000, c_x * 1000, c_y * 1000)), point_size = 40)
            pl.add_mesh(pv.Arrow(start = tvec * 1000, direction = xvec, scale='auto'), color='red')
            pl.add_mesh(pv.Arrow(start = tvec * 1000, direction = yvec, scale='auto'), color='green')
            pl.add_mesh(pv.Arrow(start = tvec * 1000, direction = zvec, scale='auto'), color='blue')
            pl.add_mesh(pv.Arrow(start = (0, 0, 0), direction = (1, 0, 0), scale='auto'), color='red')
            pl.add_mesh(pv.Arrow(start = (0, 0, 0), direction = (0, 1, 0), scale='auto'), color='green')
            pl.add_mesh(pv.Arrow(start = (0, 0, 0), direction = (0, 0, 1), scale='auto'), color='blue')
            pl.show()
            
            # Append to lists
            # Take FK matrices directly from saved pose data in step 1
            R_base2ee.append(fkMats[i][0:3,0:3])
            t_base2ee.append(fkMats[i][0:3,3])
            R_cam2obj.append(rmat)
            t_cam2obj.append(tvec)

        # Use all recorded matrices to calculate the hand-eye calibration
        print("Hand eye calibration information below.")
        calibration = cv.calibrateHandEye(R_base2ee, t_base2ee, R_cam2obj, t_cam2obj)
        print(calibration)
        R_ee2cam = calibration[0]
        t_ee2cam = calibration[1]
        
        # Verify accuracy by reprojecting points
        tf_o = []
        tf_x = []
        if method == 2:
            tf_y = []
        for i in reprojectionIndices:
            if np.any(np.isnan(circleCenters[i])):
                continue
            tf_o.append(fkMats[i][0:3, 0:3] @ (R_ee2cam @ circleCenters[i, 0, :] + t_ee2cam.T[0]) + fkMats[i][0:3,3])
            tf_x.append(fkMats[i][0:3, 0:3] @ (R_ee2cam @ circleCenters[i, 1, :] + t_ee2cam.T[0]) + fkMats[i][0:3,3])
            if method == 2:
                tf_y.append(fkMats[i][0:3, 0:3] @ (R_ee2cam @ circleCenters[i, 2, :] + t_ee2cam.T[0]) + fkMats[i][0:3,3])
        tf_o = np.array(tf_o)
        tf_x = np.array(tf_x)
        if method == 2:
            tf_y = np.array(tf_y)
        d_o = np.linalg.norm(np.repeat(np.mean(tf_o, axis = 0)[np.newaxis,:], tf_o.shape[0], axis = 0) - tf_o, axis = 1) * 1000
        d_x = np.linalg.norm(np.repeat(np.mean(tf_x, axis = 0)[np.newaxis,:], tf_x.shape[0], axis = 0) - tf_x, axis = 1) * 1000
        if method == 2:
            d_y = np.linalg.norm(np.repeat(np.mean(tf_y, axis = 0)[np.newaxis,:], tf_y.shape[0], axis = 0) - tf_y, axis = 1) * 1000
            
        plt.figure(1)
        if method == 1:
            bp = plt.boxplot([d_o, d_x], tick_labels = ["Origin Reprojection Error", "X-Axis Reprojection Error"], flierprops={'marker': 'o', 'markersize': 10, 'markerfacecolor': 'blue'})
        elif method == 2:
            bp = plt.boxplot([d_o, d_x, d_y], tick_labels = ["Origin Error", "X-Axis Error", "Y-Axis Error"])
        plt.ylabel("Error (mm)")
        plt.gca().set_ylim(bottom=0)
        for median in bp['medians']:
            median.set_color('black')
        
    else:
        print("Invalid Step. Terminating...")