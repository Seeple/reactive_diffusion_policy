'''
Unified Franka test client: Directly use Polymetis RobotInterface, no zerorpc.
Generates a test trajectory and sends it to the robot with real-time target_time.
'''
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from polymetis import RobotInterface, GripperInterface
import numpy as np
import time
import threading
from collections import deque
from loguru import logger
import enum
import torch
import scipy.spatial.transform as st

from reactive_diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from reactive_diffusion_policy.common.precise_sleep import precise_wait

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_Gripper = 3

class FrankaInterpolationController:
    def __init__(self,
                 robot_ip='0.0.0.0',
                 robot_port=50051,
                 gripper_port=50052,
                 frequency=1000):
        
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.gripper_port = gripper_port
        self.frequency = frequency
        self.control_cycle_time = 1.0 / self.frequency

        try:
            self.robot = RobotInterface(ip_address=self.robot_ip)
            logger.info(f"Connected to robot at {self.robot_ip}")
        except Exception as e:
            logger.error(f"Failed to connect to robot at {self.robot_ip}: {e}")
            raise
        # self.gripper = GripperInterface(ip_address=self.robot_ip, port=self.gripper_port)

        self.command_queue = deque(maxlen=256)
        self.pose_interp = None
        self.last_waypoint_time = None

        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])

    def get_joint_positions(self):
        try:
            return self.robot.get_joint_positions().numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to get joint positions: {e}")
            raise

    def get_ee_pose(self):
        '''
        Return:
            (x, y, z, rx, ry, rz) in flange coordinate
        '''
        try:
            data = self.robot.get_ee_pose()
            pos = data[0].numpy()
            quat_xyzw = data[1].numpy()
        except Exception as e:
            logger.error(f"Failed to get end effector pose: {e}")
            raise
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()

    def go_home(self):
        self.robot.go_home()

    def process_commands(self):
        '''
        Main loop to process commands from the queue and interpolate poses.
        '''   

        # initialize the pose interpolator
        if self.pose_interp is None:
            curr_flange_pose = self.get_ee_pose() # (x, y, z, rx, ry, rz)， in flange coordinate
            curr_time = time.monotonic()
            logger.info(f"Initial time: {curr_time} before processing commands")    
            self.pose_interp = PoseTrajectoryInterpolator(
                times=[curr_time],
                poses=[curr_flange_pose]
            )
            self.last_waypoint_time = curr_time

        logger.info("Starting Cartesian impedance control...")
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(self.Kx),
            Kxd=torch.Tensor(self.Kxd)
        )

        t_start = time.monotonic()
        iter_idx = 0

        last_print = time.monotonic()
        count = 0
        
        try:
            while True:
                t_now = time.monotonic()
                flange_pos = self.pose_interp(t_now).tolist() # (x, y, z, rx, ry, rz), in flange coordinate
                pos = torch.Tensor(flange_pos[:3])
                orientation = torch.Tensor(st.Rotation.from_rotvec(flange_pos[3:]).as_quat())
                
                # try:
                #     self.robot.update_desired_ee_pose(
                #         position=pos,
                #         orientation=orientation
                #     )
                # except Exception as e:
                #     logger.error(f"Failed to update desired end effector pose: {e}")
                #     raise

                count += 1
                if t_now - last_print > 1.0:
                    logger.info(f"update_desired_ee_pose called {count} times in last second")
                    count = 0
                    last_print = t_now

                try:
                    command = self.command_queue.popleft()
                    if command['cmd'] == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose'] # (x, y, z, rx, ry, rz), in flange coordinate
                        curr_time = t_now + self.control_cycle_time
                        target_time = command['target_time'] 

                        self.pose_interp = self.pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=self.last_waypoint_time
                        )
                        self.last_waypoint_time = target_time
                        if(target_time < curr_time):
                            logger.warning(f"Target time {target_time} is in the past, current time: {curr_time}. Skipping this waypoint.")
                        logger.info(f"New waypoint scheduled at {target_time}: {target_pose}")
                except IndexError:
                    pass

                t_wait_util = t_start + (iter_idx + 1) * self.control_cycle_time
                precise_wait(t_wait_util, time_func=time.monotonic)
                iter_idx += 1
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, stopping control loop.")
        finally:
            self.robot.terminate_current_policy()
            logger.info("Control loop terminated.")

    def generate_test_trajectory(self):
        '''
        Generate a simple trajectory: first translation, then rotation.
        Each waypoint is generated with actual current ee pose
        '''
        curr_pose = self.get_ee_pose() # (x, y, z, rx, ry, rz), in flange coordinate
        duration = 20.0
        num_points = 50

        # # Test: only add one trajectory way point
        # curr_time = time.monotonic()
        # target_pose = curr_pose.copy()
        # target_pose[0] = curr_pose[0] + 0.0005
        # target_time = curr_time + duration / num_points
        # self.command_queue.append({
        #     'cmd': Command.SCHEDULE_WAYPOINT.value,
        #     'target_pose': target_pose,
        #     'target_time': target_time
        # })


        # translation： move along x-axis for 0.05 meters 
        trans_points = 25
        curr_time = time.monotonic()
        logger.info(f"Current time: {curr_time} before generating trajectory")
        for i in range(trans_points):
            target_pose = curr_pose.copy()
            dx = (i + 1) * 0.05 / trans_points
            target_pose[0] = curr_pose[0] + dx
            target_time = curr_time + (i + 1) * duration / num_points
            self.command_queue.append({
                'cmd': Command.SCHEDULE_WAYPOINT.value,
                'target_pose': target_pose,
                'target_time': target_time
            })

        # rotation: rotation around z-axis for 0.5pi radians
        last_pose = curr_pose.copy()
        last_pose[0] = curr_pose[0] + 0.05

        for i in range(trans_points):
            target_pose = last_pose.copy()
            angle = (i + 1) * 0.5 * np.pi / trans_points
            target_pose[5] = last_pose[5] + angle
            target_time = curr_time + (i + 1 + trans_points) * duration / num_points
            self.command_queue.append({
                'cmd': Command.SCHEDULE_WAYPOINT.value,
                'target_pose': target_pose,
                'target_time': target_time
            })

    def run_trajectory_test(self):
        try:
            # current joint positions: [-0.2, -0.1, 1.5, -2.0, 0.2, 2.5, -0.4]

            print("Moving to home position...")
            self.robot.move_to_joint_positions(positions = torch.Tensor([-0.2, -0.1, 1.5, -2.0, 0.2, 2.5, -0.4]),time_to_go=5.0)
            print("Robot moved to home position.")
            
            positions = self.get_joint_positions()
            print("Current Joint Positions:", positions)
            original_ee_pose = self.get_ee_pose()
            print("Original End-Effector Pose:", original_ee_pose)

            print("Generating test trajectory...")
            self.generate_test_trajectory()
            logger.info(f"Generated {len(self.command_queue)} waypoints")

            print("Starting interpolation control thread...")

            control_thread = threading.Thread(target=self.process_commands, daemon=True)
            control_thread.start()

            # wait for the trajectory to be processed
            logger.info("Executing trajectory...")
            trajectory_duration = 25.0  
            for i in range(int(trajectory_duration)):
                logger.info(f"Trajectory execution: {i}/{int(trajectory_duration)} seconds")
                time.sleep(1)

        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            logger.exception(e)
        finally:
            logger.info("Test completed, robot policy terminated.")

if __name__ == "__main__":
    controller = FrankaInterpolationController()
    controller.run_trajectory_test()