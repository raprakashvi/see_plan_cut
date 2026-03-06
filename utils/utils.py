import pybullet as p
import numpy as np
import os
import time
import pybullet_utils_cust as pu
from itertools import groupby
from statistics import mean

def load_ur_robot(robot_initial_pose=[[0., 0., 0.], [0., 0., 0., 1.]], 
                          client_id=0,urdf_dir="./robots/urdf", urdf_file = "ur5e.urdf"):
    '''
    Function to load robot into Pybullet

    Parameters
    ----------
    robot_initial_pose : 2-element list containing XYZ position and rotation quaternion, optional
        Default rotation for the robot in Pybullet sim. The default is [[0., 0., 0.], [0., 0., 0., 1.]].
    client_id : Int, optional
        Pybullet client ID to load into. The default is 0.

    Raises
    ------
    ValueError
        Raised if file not found.

    Returns
    -------
    robot_id : Int
        Robot ID number in simulation.
    urdf_filepath : Str
        Filepath to URDF file.

    '''
    #urdf_dir = "./robots/urdf"
    #urdf_dir = "./robots/urdf"
    urdf_filepath = os.path.join(urdf_dir, urdf_file)  # Update the path to your URDF file

    if not os.path.exists(urdf_filepath):
        raise ValueError(f"URDF file {urdf_filepath} does not exist!")

    robot_id = p.loadURDF(urdf_filepath, 
                          basePosition=robot_initial_pose[0], 
                          baseOrientation=robot_initial_pose[1],
                          useFixedBase=True,
                          physicsClientId=client_id,
                          #flags=p.URDF_USE_SELF_COLLISION  # Enable self-collision checking
                        )
    return robot_id, urdf_filepath

def norm2quat(picked_normal, initial_normal = np.array([0, 0, 1])):
    '''
    Generates a quaternion from the base end effector normal (0, 0, 1) to a target vector.
    Note that a rotation with quaternion (0, 0, 0, 0) has normal vector (0, 0, 1), e.g. end-effector pointing straight upwards.
    https://math.stackexchange.com/questions/2356649/how-to-find-the-quaternion-representing-the-rotation-between-two-3-d-vectors
    
    Parameters
    ----------
    picked_normal : 3-element List
        Target vector to rotate to.

    Returns
    -------
    quaternion_IK : 4-element List [x, y, z, w]
        Quaternion for rotation.

    '''
    # Calculate quaternion for IK
    defaultNorm = initial_normal
    axis = np.cross(defaultNorm, picked_normal) / np.linalg.norm(np.cross(defaultNorm, picked_normal))
    angle = np.arctan2(np.linalg.norm(np.cross(defaultNorm, picked_normal)), np.dot(defaultNorm, picked_normal))
    quaternion_IK = [axis[0] * np.sin(angle / 2), axis[1] * np.sin(angle / 2), axis[2] * np.sin(angle / 2), np.cos(angle / 2)]
    return quaternion_IK

def hamProd(v1, v2):
    '''
    Helper function to calculate the Hamilton product (https://en.wikipedia.org/wiki/Quaternion#Hamilton_product)
    Assumed to be provided in [x, y, z, w] format. Returns in [x, y, z, w] format.
    '''
    v1 = np.array(v1)[[3, 0, 1, 2]]
    v2 = np.array(v2)[[3, 0, 1, 2]]
    return np.array([v1[0] * v2[1] + v1[1] * v2[0] + v1[2] * v2[3] - v1[3] * v2[2],
                     v1[0] * v2[2] - v1[1] * v2[3] + v1[2] * v2[0] + v1[3] * v2[1],
                     v1[0] * v2[3] + v1[1] * v2[2] - v1[2] * v2[1] + v1[3] * v2[0],
                     v1[0] * v2[0] - v1[1] * v2[1] - v1[2] * v2[2] - v1[3] * v2[3]])

def quat2norm(quat, initial_normal = np.array([0, 0, 1])):
    '''
    Function to convert given quaternion into the resulting end-effector normal vector (see norm2quat for further detail). 
    Quaternion is provided in [x, y, z, w] format.
    '''
    initial_normal = np.concatenate((initial_normal, [0]))
    quat_ = np.array([-quat[0], -quat[1], -quat[2], quat[3]])    
    return hamProd(hamProd(quat, initial_normal), quat_)[:-1]

def ur2quat(rotvec):
    '''
    Function to convert Universal Robotics rotation vector representation into quaternion. 
    See https://www.universal-robots.com/articles/ur/application-installation/explanation-on-robot-orientation/#:~:text=Rotation%20Vector,by%20a%20certain%20angle%2C%20theta. for details
    Quaternion is provided in [x, y, z, w] format.
    '''
    angle = np.linalg.norm(rotvec)
    axis = rotvec / angle # Result is unit vector
    quat = np.array([axis[0] * np.sin(angle / 2), axis[1] * np.sin(angle / 2), axis[2] * np.sin(angle / 2), np.cos(angle / 2)])
    return quat

def quat2ur(quat):
    '''
    Function to convert quaternion into Universal Robotics rotation vector representation. 
    See https://www.universal-robots.com/articles/ur/application-installation/explanation-on-robot-orientation/#:~:text=Rotation%20Vector,by%20a%20certain%20angle%2C%20theta. for details
    Quaternion should be provided in [x, y, z, w] format.
    '''
    angle = 2 * np.arccos(quat[3])
    axis = np.array([quat[0] / np.sin(angle / 2), quat[1] / np.sin(angle / 2), quat[2] / np.sin(angle / 2)])
    norm = np.linalg.norm(axis)
    if abs(norm - 1) > 0.05:
        print("Warning: Quaternion may not be a unit quaternion. Axis length is {}. Normalizing to 1...".format(norm))
    axis = axis / norm
    return axis * angle

def norm2ur(normal, initial_norm = np.array([0, 0, 1])):
    return quat2ur(norm2quat(normal, initial_norm))
    
def ur2norm(rotvec):
    return quat2norm(ur2quat(rotvec))

def tcp2rotmat(tcpPose):
    '''
    Function to convert the 6-element TCP pose from UR to a rotation matrix.
    '''
    # Quaternion to rot. matrix: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    rvec = np.array(tcpPose[3:])
    angle = np.linalg.norm(rvec)
    unitrvec = rvec / angle
    translation, rotation = tcpPose[0:3], (unitrvec[0] * np.sin(angle/2), unitrvec[1] * np.sin(angle/2), unitrvec[2] * np.sin(angle/2), np.cos(angle/2))  
    
    rotMat = np.array([[2 * (rotation[3]**2+rotation[0]**2) - 1, 2 * (rotation[0]*rotation[1]-rotation[3]*rotation[2]), 2 * (rotation[0]*rotation[2]+rotation[3]*rotation[1]), translation[0]], 
                [2 * (rotation[0]*rotation[1]+rotation[3]*rotation[2]), 2 * (rotation[3]**2+rotation[1]**2) - 1, 2 * (rotation[1]*rotation[2]-rotation[3]*rotation[0]), translation[1]],
                [2 * (rotation[0]*rotation[2]-rotation[3]*rotation[1]), 2 * (rotation[1]*rotation[2]+rotation[3]*rotation[0]), 2 * (rotation[3]**2+rotation[2]**2) - 1, translation[2]],
                [0, 0, 0, 1]])

    return rotMat

def downsampleOCT(pcd):
    '''
    Function to reduce the number of points per X/Y pair to 1 from an OCT scan. i.e. remove duplicate points via averaging
    *Currently assumes a uniformly gridded PCD with no missing entries*

    Parameters
    ----------
    pcd : Nx3 Array-like
        OCT Point cloud.
    '''
    pcd = pcd[np.lexsort((pcd[:,0], pcd[:,1]))]
    grouper = groupby(pcd, lambda x: (x[0], x[1]))
    retPCD = np.array([[key[0], key[1], mean(zi[2] for zi in z)] for key, z in grouper])
    return retPCD              
    

def IKquaternionCalc(normal, forward):
    '''
    Function to calculate the quaternion required for orientation in pybullet IK solving. 

    Parameters
    ----------
    normal : Length-3 array
        A (unit) vector denoting the Z-direction vector post-rotation. IOW, the direction of the axis of the OCT/laser. This should be pointing upwards (positive Z-component) most of the time.
    forward : Length-3 array
        A (unit) vector denoting the X-direction vector post-rotation. IOW, the direction that the robot will be "pointing" (~= to the vector between the robot base position and the robot EE position).
        This can be adjusted to change the direction that the robot is pointing to encourage smoother positions and positions further from singularity or self-collision.
        NOTE: This vector should be ~orthogonal to 'normal'

    Returns
    -------
    quat : [x, y, z, w]-style quaternion array
        Quaternion to perform rotation.

    '''
    normal = np.squeeze(normal) / np.linalg.norm(normal)
    forward = np.squeeze(forward) / np.linalg.norm(forward)
    
    y = np.cross(normal, forward)
    print(y)
    if abs(np.linalg.norm(y) - 1) > 0.1:
        print("Warning! The provided vectors to IKquaternionCalc may not be orthogonal! Strange behavior may occur in IK solving.")
    
    rotMat = np.hstack((forward[:,None], y[:,None], normal[:,None]))
    print(rotMat)
    q0 = 0.5 * np.sqrt(1 + rotMat[0,0] + rotMat[1, 1] + rotMat[2, 2])
    quat = np.array([1 / (4 * q0) * (rotMat[2, 1] - rotMat[1, 2]), 1 / (4 * q0) * (rotMat[0, 2] - rotMat[2, 0]), 1 / (4 * q0) * (rotMat[1, 0] - rotMat[0, 1]), q0])
    
    return quat

def execute_joint_trajectory(client_id, robot_id, robot, joint_trajectory, external_bodies=[], physical = False):
    """
    Executes the given joint trajectory on the robot.
    """
    marker_radius = 0.0005  # Radius of the sphere markers
    marker_color = [0, 0, 1, 1]  # Red color with full opacity

    # Create a visual shape for the sphere
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                            radius=marker_radius,
                                            rgbaColor=marker_color,
                                            physicsClientId=client_id)
    
    i = 0
    for joint_positions in joint_trajectory:
        # Check for collisions
        collision_detected = any([robot.check_external_collisions(body_id) for body_id in external_bodies])
    
        if collision_detected:
            print("Collision detected during trajectory execution. Stopping.")
            break  # Stop execution or handle as needed
        else:
            robot.control_arm_joints(joint_positions)
            eeLoc = pu.get_link_pose(robot_id, robot.LASER_LINK_INDEX, client_id=client_id)[0]
            p.createMultiBody(baseMass=0,
                                            baseVisualShapeIndex=visual_shape_id,
                                            basePosition=eeLoc,
                                            physicsClientId=client_id)
            if physical:
                robot.control_arm_joints(joint_positions, mode="phys_servo")
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1./240.)  # Adjust as needed
    
            i += 1

def robotMove(client_id, robot_id, robot, targetLink, position, normal, skipSim = False, steps = 800, returnHome = True, executeFxn = None, fxnparams = []):
    '''
    Function to move to a specified position/normal and optionally execute a specified function.

    Parameters
    ----------
    client_id : int
        Pybullet client ID
    robot_id : int
        Pybullet robot ID
    robot : UR5Controller object
        The robot object and control interface.
    targetLink : int
        ID of link to move to target position
    position : Length-3 array
        World frame position to move laser to (m).
    normal : Length-3 array
        Normal vector to calculate orientation. Note that unless upside-down, the Z-coordinate should be positive 90% of the time.
    skipSim : boolean, optional
        Whether to skip prompting and run physical scan only. The default is False.
    steps : int, optional
        Number of steps the trajectory should take. Increase to slow down. The default is 800.
    returnHome : boolean, optional
        Whether the robot should return home after ablation. The default is True.
    executeFxn : function/method
        Function to execute upon reaching target location
    fxnparams : Iterable list/array
        Params to be passed to executeFxn

    '''
    # Process Inputs
    flatPos = np.array([position[0], position[1], 0])
    forward = flatPos - np.dot(flatPos, normal) * normal
    forward = forward / np.linalg.norm(forward)
    quat = IKquaternionCalc(normal, forward)
    
    # Get initial joint values
    try:
        robot.connect_physical_robot()
        start_joints = robot.get_arm_joint_values(physical=True)
    except:
        start_joints = robot.get_arm_joint_values()
        input("Physical robot not detected! Using simulated values. Stop code and turn on robot if physical values needed.")
    robot.set_arm_joints(start_joints)

    # Solve and verify IK
    ikJoints = p.calculateInverseKinematics(robot.id,
                                               targetLink,
                                                   position,
                                                   quat,
                                                   maxNumIterations=2000,
                                                   residualThreshold=1e-5,
                                                   physicsClientId=client_id)
    within_limits = True
    arm_lowerLimits, arm_upperLimits = pu.get_joints_limits(robot_id, joints=robot.GROUP_INDEX['arm'], client_id=client_id)
    for idx, joint_value in enumerate(ikJoints):
        lower_limit = arm_lowerLimits[idx]
        upper_limit = arm_upperLimits[idx]
        if not (lower_limit <= joint_value <= upper_limit):
            print(f"Joint {idx} value {joint_value} is out of limits [{lower_limit}, {upper_limit}]")
            within_limits = False
            break
    if not within_limits:
        print("IK solution is outside joint limits.")
        #continue
    position_error, orientation_error = robot.check_position_feasibility(ikJoints, position, quat, type = ("EE" if targetLink == robot.EEF_LINK_INDEX else "EE2")) # Only supports the OCT and laser links currently
    position_tolerance = 1e-3  
    orientation_tolerance = 1e-2  
    if position_error > position_tolerance or orientation_error > orientation_tolerance:
        print("IK solution is not accurate enough.....")
        
    
    
    # Calculate trajectory and return trajectory

    end_joints = ikJoints
    trajectory = np.linspace(start_joints, end_joints, steps)
    return_trajectory = np.linspace(end_joints, start_joints, steps)
    robot.set_arm_joints(start_joints)
    
    # Execute the trajectory on the simulated robot
    if not skipSim:
        simFlag = input("Run Simulation? Y/N: ")
        if simFlag.lower() == 'y':
            print("Simulating Trajectory...")
            execute_joint_trajectory(client_id, robot_id, robot, trajectory, physical=False)
            if returnHome:
                # Execute the return trajectory
                
                input("Cut completed. Press enter to home position...")
                execute_joint_trajectory(client_id, robot_id, robot, return_trajectory, physical=False)
    
        physFlag = input("Run Physical Scan? Y/N: ")
    else:
        physFlag = 'y'
    
    # Execute the trajectory on the physical robot
    if physFlag.lower() == 'y':
        robot.set_arm_joints(start_joints)
        print("Executing Trajectory...")
        execute_joint_trajectory(client_id, robot_id, robot, trajectory, physical=True)
        
        if executeFxn != None:
            executeFxn(*fxnparams)
    
        if returnHome:
            # Execute the return trajectory
            input("Cut completed. Returning to home position...")
            execute_joint_trajectory(client_id, robot_id, robot, return_trajectory, physical=True)
            robot.control_arm_joints(start_joints, mode="phys_servo") # Extra code to try and force robot to move to exact initial position (otherwise sometimes off by ~0.1-0.2 degrees)
            print("Final Joints:", robot.get_arm_joint_values())
            print("Robot Joint Discrepancy:", np.array(robot.get_arm_joint_values()) - np.array(start_joints))
            
    robot.disconnect_physical_robot()