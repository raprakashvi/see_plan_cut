import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path
parent_dir = os.path.dirname(current_dir)
print("Parent Directory: ", parent_dir)
# Add the parent directory to sys.path
sys.path.append(current_dir)
# change the current directory to the parent directory
# os.chdir(parent_dir)

### Ur5 Setup
import os
from collections import namedtuple
import pybullet as p
import numpy as np
import pybullet_utils_cust as pu
import rtde_control
import rtde_receive

class UR5Controller(object):
    JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                         'qIndex', 'uIndex', 'flags',
                                         'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                         'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                         'parentFramePos', 'parentFrameOrn', 'parentIndex'])

    JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                           'jointReactionForces', 'appliedJointMotorTorque'])

    LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                         'localInertialFramePosition', 'localInertialFrameOrientation',
                                         'worldLinkFramePosition', 'worldLinkFrameOrientation'])

    EE2OCT = np.array([[0.01839632, -0.0326994 , -0.99929591, 0.1542934],
                           [0.01692181,  0.99933208, -0.03238906, -0.00134258],
                           [0.99968757, -0.01631405,  0.01893736, 0.04050192],
                           [0, 0, 0, 1]])
     

    # Only the arm group is defined since there's no gripper
    GROUPS = {
        'arm': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    }
    HOME = [0., -1.5, 1.0, -1.0, 0., 0.0]

    JOINT_INDICES_DICT = {}
    EE_LINK_NAME = 'ee_link'  # Ensure this matches the link name in your URDF
    LASER_LINK_NAME = 'ee2_link'  # Ensure this matches the link name in your URDF

    ARM_JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    MOVEIT_ARM_MAX_VELOCITY = [3.14] * 6  # Assuming max velocity is 3.14 rad/s for all joints

    def __init__(self, robot_id, rng=None, client_id=0, ip = None):
        self.id = robot_id
        self.rng = rng
        self.pose_rng_seed = 234
        self.initial_pose_rng = np.random.default_rng(self.pose_rng_seed)
        self.client_id = client_id
        self.initial_joint_values = self.HOME
        self.initial_base_pose = pu.get_link_pose(self.id, -1, client_id=self.client_id)
        
        # Physical parameters
        self.ip = ip
        self.ctrlInterface = None
        self.rcvInterface = None
        
        # Get joint information
        joint_infos = [p.getJointInfo(robot_id, joint_index, physicsClientId=self.client_id) \
                       for joint_index in range(p.getNumJoints(robot_id, physicsClientId=self.client_id))]
        self.JOINT_INDICES_DICT = {entry[1].decode('ascii'): entry[0] for entry in joint_infos}
        self.GROUP_INDEX = {key: [self.JOINT_INDICES_DICT[joint_name] for joint_name in self.GROUPS[key]] \
                            for key in self.GROUPS}
    
        self.EEF_LINK_INDEX = pu.link_from_name(robot_id, self.EE_LINK_NAME, client_id=self.client_id)
        self.LASER_LINK_INDEX = pu.link_from_name(robot_id, self.LASER_LINK_NAME, client_id=self.client_id)
        
        # Compute transforms
        World_2_EEF = pu.get_link_pose(self.id, self.EEF_LINK_INDEX, client_id=self.client_id)
        World_2_RobotBase = pu.get_link_pose(self.id, -1, client_id=self.client_id) # [worldLinkFramePosition, worldLinkFrameOrientation]
        self.RobotBase_2_World = p.invertTransform(World_2_RobotBase[0], World_2_RobotBase[1]) 

        self.arm_max_joint_velocities = [pu.get_max_velocity(self.id, j_id, client_id=self.client_id) for j_id in self.GROUP_INDEX['arm']]

        self.use_pybullet_ik = True

        self.reset(initial=True)
        self.reset_inner_rng()

    def connect_physical_robot(self):
        '''
        Function to connect to physical robot with control and receive interfaces. Initiates connection if first time, otherwise reconnects.

        '''
        if self.ctrlInterface == None:
            self.ctrlInterface = rtde_control.RTDEControlInterface(self.ip)
        elif self.ctrlInterface.isConnected():
            print("Control Interface Already Connected!")            
        else: 
            self.ctrlInterface.reconnect()
            
        if self.rcvInterface == None:
            self.rcvInterface = rtde_receive.RTDEReceiveInterface(self.ip)
        elif self.rcvInterface.isConnected():
            print("Receive Interface Already Connected!") 
        else:
            self.rcvInterface.reconnect()
            
    def disconnect_physical_robot(self):
        '''
        Disconnects from physical robot.
        '''
        if self.ctrlInterface == None or self.rcvInterface == None:
            print("Robot has not been connected!")
            return
        self.ctrlInterface.stopScript()
        self.ctrlInterface.disconnect()
        self.rcvInterface.disconnect()

    def reset(self, input_joint_values=None, initial=False):
        if input_joint_values is not None:
            self.start_joint_values = input_joint_values
        else:
            self.start_joint_values = self.initial_joint_values
        self.set_arm_joints(self.start_joint_values)
        self.arm_discretized_plan = None
        self.arm_wp_target_index = 0
        return self.start_joint_values

    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['arm'], joint_values, client_id=self.client_id) # set joint values instantly before starting simulation
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values, client_id=self.client_id) 

    def control_arm_joints(self, joint_values, control_type='hard', client_id=0, mode = "sim", speed = 0.2, accel = 2):
        if mode == "sim":
            pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values, control_type, client_id=self.client_id)
        else:
            if not self.ctrlInterface.isConnected():
                print("Robot not connected!")
                return
            else:
                #joint_values = deepcopy(joint_values)
                #joint_values[0] += 3*np.pi/4 #---REMOVE THIS LINE ONCE SIM ENVIRONMENT REALIGNED WITH PHYSICAL---
                if mode == "phys_servo":
                    time_start = self.ctrlInterface.initPeriod()
                    self.ctrlInterface.servoJ(joint_values, 0, 0, 0.008, 0.1, 300)
                    self.ctrlInterface.waitPeriod(time_start)
                elif mode == "phys_move":
                    self.ctrlInterface.moveJ(joint_values, speed, accel)
                else:
                    print("Mode not recognized! Select from 'sim', 'phys_servo', and 'phys_move'.")
                    return

    def get_arm_fk_pybullet(self, joint_values):
        Robot_2_EEF =  pu.forward_kinematics(self.id, 
                                     self.GROUP_INDEX['arm'], 
                                     joint_values, 
                                     self.EEF_LINK_INDEX, 
                                     client_id=self.client_id)
        World_2_EEF = p.multiplyTransforms(*self.initial_base_pose, *Robot_2_EEF)
        return World_2_EEF

    def get_arm_ik(self, World2EEF_pose, avoid_collisions=False):
        Robot_2_EEF = pu.multiply_multi_transforms(self.RobotBase_2_World, World2EEF_pose)
        jv = self.get_arm_ik_pybullet(Robot_2_EEF, avoid_collisions=avoid_collisions)
        return jv

    def get_arm_ik_pybullet(self, pose_2d, arm_joint_values=None, avoid_collisions=False):
        arm_joint_values = self.get_arm_joint_values() # get current joint values

        joint_values = p.calculateInverseKinematics(self.id,
                                                    self.EEF_LINK_INDEX,
                                                    pose_2d[0],
                                                    pose_2d[1],
                                                    currentPositions=list(arm_joint_values),
                                                    physicsClientId=self.client_id,
                                                    maxNumIterations=100)
        ik_result = list(joint_values[:6])
        ik_result = self.convert_range(ik_result) # convert to range [-pi, pi]
        return np.array(ik_result)

    def step(self, fix_turn_around=True):
        """ step the robot for 1/240 second """
        if self.arm_discretized_plan is None:
            pass
        elif self.arm_wp_target_index >= len(self.arm_discretized_plan):
            self.control_arm_joints(self.arm_discretized_plan[-1])  # stay at the last position
        else:
            if fix_turn_around:
                cur_joint = np.array(self.get_arm_joint_values())
                if (np.abs(cur_joint - np.array(self.arm_discretized_plan[self.arm_wp_target_index])) > np.pi).any():
                    self.set_arm_joints([pu.wrap_angle(jv) for jv in cur_joint])
            self.control_arm_joints(self.arm_discretized_plan[self.arm_wp_target_index])
            self.arm_wp_target_index += 1

    def get_joint_state(self, joint_index):
        return self.JointState(*p.getJointState(self.id, joint_index, physicsClientId=self.client_id))

    def get_arm_joint_values(self, convert_range=False, physical = False):
        if not physical:
            if convert_range:
                return self.convert_range([self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['arm']])
            else:
                return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['arm']]
        else:
            if not self.rcvInterface.isConnected():
                print("Robot is not connected!")
                return
            else:
                return self.rcvInterface.getActualQ()

    def get_eef_pose(self):
        return pu.get_link_pose(self.id, self.EEF_LINK_INDEX, client_id=self.client_id)
    
    @staticmethod
    def convert_range(joint_values):
        """ Convert continuous joint to range [-pi, pi] """
        circular_idx = set(range(6))
        new_joint_values = []
        for i, v in enumerate(joint_values):
            if i in circular_idx:
                new_joint_values.append(pu.wrap_angle(v))
            else:
                new_joint_values.append(v)
        return new_joint_values

    def reset_inner_rng(self):
        """ Reset the random number generator for the initial pose. """
        self.initial_pose_rng = np.random.default_rng(self.pose_rng_seed)

    def check_position_feasibility(self, joint_positions=None, wp=None, ee_orientation=None, type = 'EE'):
        """ Check if the given position is feasible for the robot. """
        if joint_positions is not None:
            original_joint_positions = self.get_arm_joint_values()
            try:
                self.set_arm_joints(joint_positions)
                p.stepSimulation(physicsClientId=self.client_id)
                if type == 'EE':
                    link_state = pu.get_link_state(self.id, self.EEF_LINK_INDEX, client_id=self.client_id)
                elif type == 'EE2':
                    link_state = pu.get_link_state(self.id, self.LASER_LINK_INDEX, client_id=self.client_id)
                actual_ee_position = link_state.linkWorldPosition
                actual_ee_orientation = link_state.linkWorldOrientation

                position_error = np.linalg.norm(np.array(actual_ee_position) - np.array(wp))
                orientation_diff = p.getDifferenceQuaternion(actual_ee_orientation, ee_orientation)
                w = np.clip(abs(orientation_diff[3]), 0.0, 1.0)  # Correct clamping
                orientation_error = 2 * np.arccos(w)
                print(f"Position error: {position_error}, Orientation error: {orientation_error}")

                return position_error, orientation_error
            finally:
                # Restore original joint positions after feasibility check
                self.set_arm_joints(original_joint_positions)
                p.stepSimulation(physicsClientId=self.client_id)

    def check_external_collisions(self, target_body_id, distance_threshold=0.01):
        """
        Check if the robot is colliding with any external body.
        
        Args:
            target_body_id (int): The body ID of the external object (e.g., table or cube).
            distance_threshold (float): The distance threshold for detecting collisions.
            
        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        num_robot_links = p.getNumJoints(self.id, physicsClientId=self.client_id)

        # Check all robot links against the external body
        for link_idx in range(-1, num_robot_links):  # Include the base (-1) link
            closest_points = p.getClosestPoints(bodyA=self.id, bodyB=target_body_id,
                                                linkIndexA=link_idx, distance=distance_threshold,
                                                physicsClientId=self.client_id)
            if len(closest_points) > 0:
                print(f"Collision detected between robot link {link_idx} and external body {target_body_id}")
                return True  # Return True if any collision is detected
        return False