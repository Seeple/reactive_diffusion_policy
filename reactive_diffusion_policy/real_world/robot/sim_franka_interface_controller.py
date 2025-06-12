'''
This script launch a Open3D visualizer 
For debugging pose trajectory interpolator in franka_server
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import zerorpc
import scipy.spatial.transform as st
import threading
import time
from loguru import logger

class SimFrankaServer:
    def __init__(self, 
                 server_ip="192.168.2.187", 
                 server_port=5000):
        """
        Initialize the visualizer and connect to the Franka server.
        """
        self.tcp_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # 7D TCP (x, y, z, qw, qx, qy, qz)
        self.gripper_width = 0.05  
        self.lock = threading.Lock()  # thread lock for synchronization
        # self.update_event = threading.Event()
        
    def tip_pose_visualize(self):
        """
        Visualize the virtual robot's TCP pose in a 3D plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # tcp position
        point, = ax.plot([0], [0], [0], 'ro')
        # # tcp orientation
        # ref_pos = np.array([-0.4, 0.4, 0.4])
        # ax.quiver(ref_pos[0], ref_pos[1], ref_pos[2],
        #      0.1, 0, 0, color='r', length=0.1, normalize=True)  # X
        # ax.quiver(ref_pos[0], ref_pos[1], ref_pos[2],
        #      0, 0.1, 0, color='g', length=0.1, normalize=True)  # Y
        # ax.quiver(ref_pos[0], ref_pos[1], ref_pos[2],
        #      0, 0, 0.1, color='b', length=0.1, normalize=True)  # Z
        
        # ax.text(ref_pos[0]+0.15, ref_pos[1], ref_pos[2], 'X', color='red')
        # ax.text(ref_pos[0], ref_pos[1]+0.15, ref_pos[2], 'Y', color='green')
        # ax.text(ref_pos[0], ref_pos[1], ref_pos[2]+0.15, 'Z', color='blue')
        
        orient_pos = np.array([-0.4, 0.4, 0.4])
        tcp_arrows = []
        while True:
            # self.update_event.wait()  # blocking
            # self.update_event.clear()
            
            with self.lock:
                tcp_position = self.tcp_pose[:3]
                tcp_orientation = self.tcp_pose[3:] # (4) (qw, qx, qy, qz)
                print(tcp_orientation, tcp_position)
                
            rotation_matrix = st.Rotation.from_quat(tcp_orientation).as_matrix()
            
            x_axis = rotation_matrix[:, 0]  
            y_axis = rotation_matrix[:, 1]  
            z_axis = rotation_matrix[:, 2] 
             
            # update coordiante axes
            point.set_data([tcp_position[0]], [tcp_position[1]])
            point.set_3d_properties([tcp_position[2]])
            
            # clear TCP coordinate in last frame
            for arrow in tcp_arrows:
                arrow.remove()
            tcp_arrows.clear()
            
            tcp_arrows.append(ax.quiver(orient_pos[0], orient_pos[1], orient_pos[2],
                x_axis[0], x_axis[1], x_axis[2], color='r', length=0.2, normalize=True))  # x
            tcp_arrows.append(ax.quiver(orient_pos[0], orient_pos[1], orient_pos[2],
                y_axis[0], y_axis[1], y_axis[2], color='g', length=0.2, normalize=True))  # y
            tcp_arrows.append(ax.quiver(orient_pos[0], orient_pos[1], orient_pos[2],
                z_axis[0], z_axis[1], z_axis[2], color='b', length=0.2, normalize=True))  # z
            
            plt.draw()
            plt.pause(0.1)
            
    def get_current_tcp(self):
        """
        Get the current TCP pose of the robot.
        """
        return self.tcp_pose.tolist() # (7) (x, y, z, qw, qx, qy, qz), flange coordinate   
    
    def get_robot_state(self):
        """
        Get the current state of the virtual robot.
        """
        with self.lock:
            return {
                "leftRobotTCP": self.get_current_tcp(),
                "leftGripperState": [self.gripper_width, 0]
            }
            
    def get_ee_pose(self):
        pos = self.tcp_pose[:3]
        quat_xyzw = self.tcp_pose[3:]
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist() # (6) (x, y, z, rx, ry, rz), flange coordiate
    
    def update_desired_ee_pose(self, pose):
        pose = np.asarray(pose) # （6） (x, y, z, rx, ry, rz)
        with self.lock:
            self.tcp_pose[:3] = pose[:3]
            self.tcp_pose[3:] = st.Rotation.from_rotvec(pose[3:]).as_quat()
        # self.update_event.set()
        
    def move_gripper(self, width, velocity, force_limit):
        """
        Move the gripper to a target width with specified velocity and force limit.
        """
        with self.lock:
            self.gripper_width = width
        # self.update_event.set()
     
    def move_gripper_force(self, force_limit):
        """
        Close the gripper with a specified force limit.
        """
        with self.lock:
            self.gripper_width = 0.0  # close the gripper
        # self.update_event.set()
    
if __name__ == "__main__":
    sim_franka_server = SimFrankaServer()

    server = zerorpc.Server(sim_franka_server)
    server.bind("tcp://192.168.2.187:5000")
    logger.debug("SimFrankaServer is running on tcp://192.168.2.187:5000")
    
    visualization_thread = threading.Thread(target=sim_franka_server.tip_pose_visualize, daemon=True)
    visualization_thread.start()
    server.run()
         
        
