'''
This script is the client scipt to test the whole Network connection
between the Franka hardware, the NUC and the Desktop.
'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import zerorpc
import numpy as np
import time
import threading
from collections import deque
from loguru import logger
import enum
from reactive_diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from reactive_diffusion_policy.common.precise_sleep import precise_sleep, precise_wait

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_Gripper = 3

class FrankaInterpolationController:
    def __init__(self, 
                 robot_ip='192.168.2.216', 
                 robot_port='4242', 
                 gripper_ip='192.168.2.216',
                 gripper_port='4241',
                 host_ip='192.168.2.240', 
                 port=8092, 
                 Kx_scale=1.0,
                 Kxd_scale=1.0,
                 frequency=1000):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        
        self.gripper_ip = gripper_ip
        self.gripper_port = gripper_port
        
        self.host_ip = host_ip
        self.port = port
        
        self.frequency = frequency
        self.control_cycle_time = 1.0 / self.frequency
        
        self.pose_queue = deque(maxlen=256) # To store the interpolated tip poses
        
        self.robot_client = zerorpc.Client()
        print(f"Connecting to robot at {self.robot_ip}:{self.robot_port}....")
        self.robot_client.connect(f"tcp://{self.robot_ip}:{self.robot_port}")
        print(f"Successfully connect to robot server {self.robot_ip}:{self.robot_port}....")
        
        # initialize interpolator related varibales
        self.command_queue = deque(maxlen=256)
        self.pose_interp = None
        self.last_waypoint_time = None
        self.should_stop = False
        
    def get_joint_positions(self):
        """
        Get the current joint positions of the robot.
        """
        try:
            return self.robot_client.get_joint_positions()
        except Exception as e:
            print("Error getting joint positions:", e)
            
    def get_ee_pose(self):
        """
        Get the current end-effector pose of the robot.
        """
        try:
            return self.robot_client.get_ee_pose()
        except Exception as e:
            print("Error getting end effector pos:", e)
    
    def move_to_joint_positions(self, positions, time_to_go):
        '''
        Move the robot to specified joint positions over a given time duration.
        '''
        try:
            self.robot_client.move_to_joint_positions(positions, time_to_go)
        except Exception as e:
            print("Error moving to joint positions:", e)
    
    def go_home(self):
        '''
        Move the robot to the home position.
        '''
        try:
            self.robot_client.go_home()
        except Exception as e:
            print("Error moving to home position:", e)
            
    def stop_client(self):
        """
        Stop the client connection.
        """
        try:
            self.robot_client.close()
            print("Client connection closed.")
        except Exception as e:
            print("Error closing client:", e)
            
    def basic_test(self):
        """
        Run a basic test to check the connection and functionality.
        """
        try:
            print("=== Basic Test ===")
            
            print("\n=== Joint Positions Test ===")
            joint_positions = self.get_joint_positions()
            print("Joint Positions:", joint_positions)
            
            print("\n=== EE Pose Test ===")
            ee_pose = self.get_ee_pose()
            print("End-Effector Pose:", ee_pose)
            
            print("\n=== Go home Test ===")
            self.go_home()
            print("Robot moved to home position.")
            positions = self.get_joint_positions()
            print("Joint Positions after homing:", positions)
            
        except Exception as e:
            print("Basic test failed:", e)
            
        finally:
            self.stop_client()
       
    def process_commands(self):
        '''
        Main loop to process commands from the queue and interpolate poses.
        '''   
        # initialize the pose interpolator
        if self.pose_interp is None:
            try:
                # curr_flange_pose = np.array(self.get_ee_pose())
                curr_flange_pose = np.array([0.5, 0.0, 0.5, 0.5, 0.5, 0.5])  # (6) (x, y, z, rx, ry, rz), in flange coordinate
                curr_time = time.monotonic()
                self.pose_interp = PoseTrajectoryInterpolator(
                    times=[curr_time],
                    poses=[curr_flange_pose]
                )
                self.last_waypoint_time = curr_time
            except Exception as e:
                logger.error(f"Failed to initialize interpolator: {e}")
                return
            
        # main loop to process commands
        t_start = time.monotonic()
        iter_idx = 0
        
        while not self.should_stop:
            try:
                t_now = time.monotonic()
                
                # obtain the interpolated tip pose at the current time
                flange_pose = self.pose_interp(t_now)
                self.pose_queue.append(flange_pose.tolist())  # store the interpolated pose

                # process new target poses in the command queue
                try:
                    command = self.command_queue.popleft()
                    if command['cmd'] == Command.SCHEDULE_WAYPOINT.value: # schedule a new waypoint
                        target_pose = command['target_pose']
                        curr_time = t_now + self.control_cycle_time
                        target_time = command['target_time']

                        self.pose_interp = self.pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=self.last_waypoint_time
                        )
                        self.last_waypoint_time = target_time
                        logger.info(f"New waypoint scheduled at {target_time}: {target_pose}")
                except IndexError:
                    pass
            
                # control frequency of the control command
                t_wait_util = t_start + (iter_idx + 1) * self.control_cycle_time
                precise_wait(t_wait_util, time_func=time.monotonic)
                iter_idx += 1
                
            except Exception as e:
                logger.error(f"Error in process_commands: {e}")
                break
    
    # run zerorpc client in the main thread        
    def send_poses(self):
        last_print_time = time.time()
        pose_counter = 0
        
        while not self.should_stop:
            try:
                if len(self.pose_queue) > 0:
                    pose = self.pose_queue.popleft()
                    self.robot_client.update_desired_ee_pose(pose)
                    pose_counter += 1
                    
                    current_time = time.time()
                    if current_time - last_print_time >= 1.0:
                        logger.info(f"Sent {pose_counter} poses in the last second")
                        logger.info(f"Current pose: {pose}")
                        pose_counter = 0
                        last_print_time = current_time
                        
            except Exception as e:
                logger.error(f"Error sending pose: {e}")
            # time.sleep(0.0005)
            
    def generate_test_trajectory(self):
        '''
        Generate a simple traejctory to test functionality.
        '''
        curr_pose = np.array(self.get_ee_pose()) 
        print(f"Current EE Pose in generate_test_trajectory: {curr_pose}")
        duration = 10.0
        num_points = 50
        
        # transition
        curr_time = time.monotonic()
        for i in range(num_points):
            target_pose = curr_pose.copy()
            dx = 0.2 * np.sin(2 * np.pi * i / num_points)
            target_pose[0] = curr_pose[0] + dx
            
            target_time = curr_time + (i + 1) * duration / num_points
            
            self.command_queue.append({
                'cmd': Command.SCHEDULE_WAYPOINT.value,
                'target_pose': target_pose,
                'target_time': target_time
            })
        
            
    def run_trajectory_test(self):
        """
        Run a trajectory test by generating a circular trajectory and processing commands.
        """
        try: 
            # go home
            print("Moving to home position...")
            self.go_home()
            print("Robot moved to home position.")
            positions = self.get_joint_positions()
            print("Current Joint Positions:", positions)
            
            # start impedence control
            print("Starting impedance control....")
            Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
            Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])
            self.robot_client.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())
            
            print("Generating test trajectory...")
            self.generate_test_trajectory()
            logger.info(f"Generated {len(self.command_queue)} waypoints")
            
            print("Starting interpolation control thread...")
            interpolation_thread = threading.Thread(target=self.process_commands, daemon=True)
            interpolation_thread.start()
            
            print("Starting pose sending in main thread...")
            self.send_poses()
                
        except Exception as e:
            print(f"Test failed with error: {e}")
            logger.exception(e)
        finally:
            print("Process finished, stopping the robot client...")
            self.robot_client.close()
            print("Test completed.")
        
if __name__ == "__main__":
    controller = FrankaInterpolationController()
    # controller.basic_test()
    controller.run_trajectory_test()