import pybullet as p
import sys
sys.path.append("./")
sys.path.append("./utils")
import numpy as np
import pybullet_data
import utils
import time
from UR5Controller import UR5Controller
from ndyag_control.laser_control_pwm import robot_laser_cut_control
    
def plane(x, A, B, C):
    # Helper function to perform planar fit
    return A + B*x[0] + C*x[1]

def laserCut(dt, frequency, dutycycle):
    # Execute laser cut
    time.sleep(2)
    print("Executing Laser Cut...")    
    try:
        robot_laser_cut_control(dt, frequency, dutycycle)
    except Exception as e:
        print(f"Laser cutting failed with error: {e}")
        raise e
 
def scanOCT(octObj, filename):
    # Execute OCT scan
    flag = octObj.start_oct_scan(filename)
    if flag:
        return
    else:
        print("OCT Scan Failed.")
        return
        
#%% Main
if __name__ == "__main__":

    #--------------------MAKE SURE I'M CORRECT!-------------------------    
    # Laser parameters
    time_period = 1.5
    frequency = 100
    dutycycle = 0.90
    
    homePosition = np.array([92.25, -59.66, 73.01, -194.21, -65.34, 88.00]) * np.pi / 180  # Joint angles for home position for the robot to return to for lens cleaning (radians)
    coefs = np.array([-9.28121650e-01, -8.55212697e-04,  4.75074654e-02]) # Coefficients for the plane representing the tissue in the OCT scan. Used to determine an OCT to tissue transform, as plan inputs are generated in the tissue frame.
    # FK Matrix for the initial pose where the first OCT scan was conducted (used for later transforms). (XYZ, rx ry rz values from touch pendant)
    fkMat = utils.tcp2rotmat(np.array([0.43627, -0.55389, 0.25745, 0.739, 1.442, -0.781]))

    inputSeq = np.array([[0, 0, 0, 0, dutycycle]]) # Input sequence with entries of the form [xPos, yPos, xAngle, yAngle, dutyCycle]
    #--------------------MAKE SURE I'M CORRECT!-------------------------

    # Calculate a new transform between the OCT frame and the "tisue frame" where all the cuts were planned
    x0 = 0.003577 # Half-length of OCT volume in X-direction (m)
    y0 = -0.00357 # Half-length of OCT volume in Y-direction (m)
    z0 = plane([x0 * 1000, y0 * 1000], *coefs) / 1000
    
    normal = np.array([-coefs[1], -coefs[2], 1.0]) / np.linalg.norm([-coefs[1], -coefs[2], 1.0]) # Normal vector at point to cut
    xAxis = np.array([1, 0, 0]) - np.dot(np.array([1, 0, 0]), normal) * normal
    xAxis = xAxis / np.linalg.norm(xAxis)
    zAxis = normal
    yAxis = np.cross(zAxis, xAxis)
    
    rmat = np.hstack((xAxis[:,None], yAxis[:,None], zAxis[:,None])) # Rotation matrix from OCT frame to tissue frame
    tvec = np.array([x0, y0, z0]) # Translation vector from OCT frame to tissue frame
    Hmat = np.array([[*rmat[0], tvec[0]],
                     [*rmat[1], tvec[1]],
                     [*rmat[2], tvec[2]],
                     [0, 0, 0, 1]])
                    
    #%% Robot Loading and Vis.
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)

    robot_id, urdf_path = utils.load_ur_robot(robot_initial_pose=[[0, 0, 0], [0, 0, 0, 1]], client_id=client_id, urdf_dir = "robots/urdf", urdf_file = "ur5e_fixed.urdf")
    robot = UR5Controller(robot_id, rng=None, client_id=client_id, ip="192.168.1.103")
    EE2OCT = robot.EE2OCT
    
    robot.connect_physical_robot()

    for ind, i in enumerate(inputSeq):
        if (i[4] > 1) or (i[4] < 0.1):
            print("Duty cycle {} not valid! Should be between 0.1 or 1".format(i[4]))
            continue
        
        input("Executing cut {}/{}...".format(ind + 1, inputSeq.shape[0]))

        tx = i[2]
        ty = i[3]
        
        # Convert target point and angle from tissue frame to world frame
        pointTissueF = np.array([i[0], i[1], 0]) / 1000
        normalTissueF = -np.array([-np.sin(ty), np.sin(tx) * np.cos(ty), -np.cos(tx) * np.cos(ty)])
        pointsWorld = (fkMat @ EE2OCT @ Hmat @ np.array([pointTissueF[0], pointTissueF[1], pointTissueF[2], 1]))[0:3]
        normalWorld = (fkMat @ EE2OCT @ Hmat @ np.array([normalTissueF[0], normalTissueF[1], normalTissueF[2], 0]))[0:3]

        success = 0
        while success == 0:
            try:
                utils.robotMove(client_id, robot_id, robot, robot.LASER_LINK_INDEX, pointsWorld, normalWorld, skipSim = True, steps = 200, returnHome = False, 
                    executeFxn=laserCut, fxnparams = [time_period, frequency, i[4] * 100])
                success = 1
            except Exception as e:
                input(f"Error during robot move or laser cut: {e}. Press enter to retry...")
                time.sleep(2)
        
        # Code to pause cutting and return to home, for lens cleaning and/or other intermediate tasks
        if (ind + 1) % 5 == 0 and (ind + 1) != inputSeq.shape[0]:
            robot.connect_physical_robot()
            robot.control_arm_joints(homePosition, mode = 'phys_move', speed = 0.1, accel = 2)
            robot.disconnect_physical_robot()
            input("Clean lens...")
