import pybullet as p
import sys
sys.path.append("./")
sys.path.append("./utils")
import numpy as np
import pybullet_data
import utils.utils
from utils.UR5Controller import UR5Controller
from ndyag_control.laser_control_pwm import robot_laser_cut_control
    
def laserCut(time, frequency, dc):
    '''
    Function to execute a laser cut on 1060nm NdYAG laser with provided cut time, frequency, and duty cycle
    '''
    print("Executing Laser Cut...")    
    try:
        robot_laser_cut_control(time, frequency, dc)
    except Exception as e:
        print(f"Laser cutting failed with error: {e}")
        
#%% Main
if __name__ == "__main__":
    
    #--------------------MAKE SURE I'M CORRECT!------------------------- 
    ip = "192.168.1.103" # Robot IP Address
    # NOTE: These values are calculated and fed to the script. Can be automated in the future.
    fkMat = utils.tcp2rotmat([0.48460, -0.47550, 0.23663, 0.628, 1.445, -0.603]) # UR5 tool end x, y, z translation and rx, ry, rz rotation (https://www.universal-robots.com/articles/ur/application-installation/explanation-on-robot-orientation/) at time of OCT scan.
    points = np.array([[5.101055838856132,-1.3280734232621363,-3.6181454821356525]]) / 1000 # Point to cut in the OCT frame of reference (mm)
    normal = np.array([[0.01947261375155126, 0.008660594385457576, 0.9997728799175208]]) # Normal unit vector at point to cut in the OCT frame of reference
    time = 0.5 # Laser ablation pulse time (s)
    dutyCycle = 70 # Laser PWM duty cycle
    #--------------------MAKE SURE I'M CORRECT!-------------------------

    #%% Robot Loading and Vis.
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)

    robot_id, urdf_path = utils.load_ur_robot(robot_initial_pose=[[0, 0, 0], [0, 0, 0, 1]], client_id=client_id, urdf_dir = "robots/urdf", urdf_file = "ur5e_fixed.urdf")
    robot = UR5Controller(robot_id, rng=None, client_id=client_id, ip=ip)
    
    EE2OCT = robot.EE2OCT
    
    robot.connect_physical_robot()
    
    # Convert points to world/robot base frame
    pointsWorld = (fkMat @ EE2OCT @ np.array([points[0][0], points[0][1], points[0][2], 1]))[0:3]
    normalWorld = (fkMat @ EE2OCT @ np.array([normal[0][0], normal[0][1], normal[0][2], 0]))[0:3]
    
    # Move and perform ablation
    utils.robotMove(client_id, robot_id, robot, robot.LASER_LINK_INDEX, pointsWorld, normalWorld, steps = 100, returnHome = True, executeFxn=laserCut, fxnparams = [time, 100, dutyCycle])
