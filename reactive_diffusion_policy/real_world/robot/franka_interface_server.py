'''
Franka Interface server, launch plymetis on NUC
RobotServer part of BimanualFlexivServer
'''

import zerorpc
from polymetis import RobotInterface, GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch

class FrankaInterface:
    def __init__(self):
        '''
        RobotInterface is a wrapper around the Franka Control Interface (FCI)
        and provides a high-level API for controlling the robot.
        '''
        # TODO: Replace 'localhost' and port with the actual IP address and port of the robot and gripper
        self.robot = RobotInterface(ip_address='localhost', port=50051)
        self.gripper = GripperInterface(ip_address='localhost', port=50052)
        
    def clear_fault(self):
        """
        Clear any fault in the robot.
        """
        self.robot.clear_fault()
        
    def get_current_tcp(self):
        """
        Get the current TCP pose of the robot.
        """
        return self.get_ee_pose()

    # libfranka Gripper State has no element 'gripper_force'
    
    def get_robot_state(self):
        """
        Get the current state of the robot.
        """
        robot_state = self.robot.get_robot_state()
        gripper_state = self.gripper.get_state()
        ee_pose = self.get_ee_pose()
        joint_positions = self.get_joint_positions()
        joint_velocities = self.get_joint_velocities()
        return {
            "leftRobotTCP": ee_pose, # (7) (x, y, z, qw, qx, qy, qz)
            # TODO: Obtaib the TCP velocity (6) (vx, vy, vz, wx, wy, wz)
            # "leftRobotTCPVel": joint_velocities, # (6) (vx, vy, vz, wx, wy, wz)
            # TODO: check whether TCP wrench can be obtained from following API
            "leftRobotTCPWrench": robot_state.motor_torques_external.tolist(), # (6) (fx, fy, fz, mx, my, mz)
            # TODO: Obtaib gripper force
            "leftGripperState": [gripper_state.width, 0] # (2) (width, force)
        }

    def get_ee_pose(self):
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()
    
    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()
    
    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()
    
    def move_to_joint_positions(self, positions, time_to_go):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go
        )
    
    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx),
            Kxd=torch.Tensor(Kxd)
        )

    def update_desired_ee_pose(self, pose):
        pose = np.asarray(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat())
        )
        
    def move_gripper(self, width, velocity, force_limit):
        """
        Move the gripper to a target width with specified velocity and force limit.
        """
        self.gripper.goto(width=width, speed=velocity, force=force_limit)

    def move_gripper_force(self, force_limit):
        """
        Close the gripper with a specified force limit.
        """
        self.gripper.grasp(force=force_limit)

    def stop_gripper(self):
        """
        Stop the gripper's current motion.
        """
        self.gripper.stop()

    def terminate_current_policy(self):
        self.robot.terminate_current_policy()

s = zerorpc.Server(FrankaInterface())
s.bind("tcp://0.0.0.0:4242")
s.run()