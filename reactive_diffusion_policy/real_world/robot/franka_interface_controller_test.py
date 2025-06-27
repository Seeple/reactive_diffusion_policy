'''
A simple controller to test connection between the Franka hardware and the Desktop
'''

import zerorpc
from polymetis import RobotInterface
from reactive_diffusion_policy.common.space_utils import pose_6d_to_pose_7d
import numpy as np
import scipy.spatial.transform as st
import torch

class FrankaInterfaceControllerTest:
    def __init__(self):
        """
        RobotInterface is a wrapper around the Franka Control Interface (FCI)
        and provides a high-level API for controlling the robot.
        """
        self.robot = RobotInterface(ip_address='localhost', port=50051)
        
    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()
    
    def get_current_tcp(self):
        return pose_6d_to_pose_7d(np.asarray(self.get_ee_pose())).tolist() # (7) (x, y, z, qw, qx, qy, qz), in flange coordinate
    
    def get_ee_pose(self):
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist() # (6) (x, y, z, rx, ry, rz), in flange coordinate
    
    def move_to_joint_positions(self, positions, time_to_go):
        '''
        Move the robot to specified joint positions over a given time duration.
        '''
        try:
            positions_tensor = torch.tensor(positions, dtype=torch.float32)
            self.robot.move_to_joint_positions(positions_tensor, time_to_go)
        except Exception as e:
            print("Error moving to joint positions:", e)
            
    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx),
            Kxd=torch.Tensor(Kxd)
        )
            
    def update_desired_ee_pose(self, pose): # pose: (6) (x, y, z, rx, ry, rz)
        """
        Update the desired end-effector pose.
        """
        try:
            self.robot.update_desired_ee_pose(position=torch.Tensor(pose[:3]), orientation=torch.Tensor(pose[3:]))
        except Exception as e:
            print("Error updating desired end-effector pose:", e)
    
    def go_home(self):
        """
        Move the robot to the home position.
        """
        self.robot.go_home()
 
s = zerorpc.Server(FrankaInterfaceControllerTest())
s.bind("tcp://10.53.21.79:4242")

print("Starting FrankaInterfaceControllerTest server...")
try:
    s.run()
except KeyboardInterrupt:
    print("Server stopped by user.")
finally:
    s.close()
    print("Server closed.")  