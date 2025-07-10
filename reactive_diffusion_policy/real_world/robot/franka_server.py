'''
Franka Interface Client and Interpolator:
Recieve commands from teleoperation server,
Interpolate the moving trajectory,
and Send commands to Franka through directly using Franka Control Interface (FCI).
Expose the base controller through FASTAPI.
'''

import threading
import time
import numpy as np
import torch
import enum
from fastapi import FastAPI, HTTPException
import scipy.spatial.transform as st
from loguru import logger
from typing import Dict
from collections import deque
import uvicorn
import argparse
import multiprocessing as mp

from polymetis import RobotInterface, GripperInterface

from reactive_diffusion_policy.common.data_models import (TargetTCPRequest, MoveGripperRequest, BimanualRobotStates)
from reactive_diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from reactive_diffusion_policy.common.precise_sleep import precise_sleep, precise_wait

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_GRIPPER = 3

class FrankaServer:
    def __init__(self, 
                 # TODO: Replace with the actual IP address of the robot and gripper
                 robot_ip='0.0.0.0', 
                 robot_port=50051, 
                 gripper_port=50052,
                 host_ip='0.0.0.0', 
                 port=8092, 
                 sim_server_ip='0.0.0.0',
                 sim_server_port=5000,
                 Kx_scale=1.0,
                 Kxd_scale=1.0,
                 frequency=1000):
        """
        robot ip: ip address of NUC or the Robot simulation server
        host_ip: ip address which runs the FastAPI server
        sim_ip: ip address of the simulator, used for testing
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains.
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.gripper_port = gripper_port
        self.host_ip = host_ip
        self.sim_server_ip = sim_server_ip
        self.sim_server_port = sim_server_port
        self.port = port
        self.frequency = frequency
        self.control_cycle_time = 1.0 / self.frequency

        # Initialize the robot and gripper interfaces
        self.robot = RobotInterface(ip_address=self.robot_ip)
        # self.robot = RobotInterface(ip_address=self.robot_ip, port=self.robot_port)
        # self.gripper = GripperInterface(ip_address=self.robot_ip, port=self.gripper_port)
        
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale

        self.command_queue = deque(maxlen=256)
        self.pose_interp = None
        self.last_waypoint_time = None
        
        self.app = FastAPI()
        self.setup_routes()
        
        # TODO: check whether we need to initialize different RobotInterface for each request
        # self.client_mapping = {}


    def setup_routes(self):
        @self.app.post('/clear_fault')
        async def clear_fault():
            """
            Clear any fault in the robot.
            """
            logger.warning("Fault occurred on franka robot server, trying to clear ...")
            try:
                self.robot.clear_fault()
            except Exception as e:
                logger.error(f"Failed to clear fault: {e}")
            return {"message": "Fault cleared"}

        @self.app.get('/get_current_tcp/{robot_side}')
        async def get_current_tcp(robot_side: str):
            """
            Get the current TCP pose of the robot.
            Returns:
                (x, y, z, qw, qx, qy, qz), in flange coordinate
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            try:
                cur_tcp = self.get_current_tcp()
                # logger.info(f"Current TCP: {cur_tcp}")
            except Exception as e:
                logger.error(f"Failed to get current TCP: {e}")
                raise HTTPException(status_code=500, detail="Failed to get current TCP")
            return cur_tcp

        @self.app.get('/get_current_robot_states')
        async def get_current_robot_state() -> BimanualRobotStates:
            """
            Get the current state of the robot.
            """
            try:
                state = self.get_robot_state()
                logger.info(f"Current Robot State: {state}")
            except Exception as e:
                logger.error(f"Failed to get robot state: {e}")
                raise HTTPException(status_code=500, detail="Failed to get robot state")
            return BimanualRobotStates(**state)

        # Franka gripper command is non-realtime and low-frequency
        # Thus don't need to interpolate the gripper command 
        @self.app.post('/move_gripper/{robot_side}')
        async def move_gripper(robot_side: str, request: MoveGripperRequest)-> Dict[str, str]:
            """
            Move the gripper to a target width with specified velocity and force limit.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            try:
                self.gripper.goto(width=request.width, speed=request.velocity, 
                                  force=request.force_limit)
                logger.info(f"Gripper moving to width {request.width} with velocity {request.velocity} and force limit {request.force_limit}")
            except Exception as e:
                logger.error(f"Failed to move gripper: {e}")
                raise HTTPException(status_code=500, detail="Failed to move gripper")
            return {"message": f"Gripper moving to width {request.width}"}

        @self.app.post('/move_gripper_force/{robot_side}')
        async def move_gripper_force(robot_side: str, request: MoveGripperRequest)-> Dict[str, str]:
            """
            Close the gripper with a specified force limit.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            try:
                self.gripper.grasp(force=request.force_limit)
                logger.info(f"Gripper grasping with force {request.force_limit}")
            except Exception as e:
                logger.error(f"Failed to move gripper: {e}")
                raise HTTPException(status_code=500, detail="Failed to move gripper")
            return {"message": f"Gripper grasping with force {request.force_limit}"}

        @self.app.post('/stop_gripper/{robot_side}')
        async def stop_gripper(robot_side: str)-> Dict[str, str]:
            """
            Stop the gripper's current motion.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            try:
                self.gripper.stop()
                logger.info("Gripper stopped")
            except Exception as e:
                logger.error(f"Failed to stop gripper: {e}")
                raise HTTPException(status_code=500, detail="Failed to stop gripper")
            return {"message": "Gripper stopped"}

        @self.app.post('/move_tcp/{robot_side}')
        async def move_tcp(robot_side: str, request: TargetTCPRequest):
            '''
            Move the robot to a target TCP pose.
            Add a new low-frequency target pose to the command queue, waiting for the interpolator to process it.
            '''
            if robot_side != "left":
                logger.info("Only left arm is supported")
                
            target_7d_pose = np.array(request.target_tcp) # (x, y, z, qw, qx, qy, qz), in flange coordinate
            # target_pose = pose_7d_to_pose_6d(target_7d_pose) # (x, y, z, r, p, y)， in flange coordinate
            pos = target_7d_pose[:3]
            quat_wxyz = target_7d_pose[3:]
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]] # wxyz to xyzw
            rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
            target_pose = np.concatenate([pos, rotvec]) # (x, y, z, rx, ry, rz)， in flange coordinate

            
            curr_time = time.monotonic()
            command_duration = 1/90 # 90Hz for high-level control frequency
            target_time = curr_time + command_duration
            self.command_queue.append({
                'cmd': Command.SCHEDULE_WAYPOINT.value,
                'target_pose': target_pose,
                'target_time': target_time
            })
            # logger.info("New command added into command queue!")
            
            return {"message": "Waypoint added for franka robot"}

        @self.app.post('/birobot_go_home')
        async def go_home():
            """
            Move the robot to its home position.
            """
            logger.info("Moving Franka robot to its home position...")
            try:
                self.go_home()
            except Exception as e:
                logger.error(f"Failed to move robot to home position: {e}")
                raise HTTPException(status_code=500, detail="Failed to move robot to home position")
            return {"message": "Robot moved to home position"}
        
        

    def get_current_tcp(self):
        """
        Get the current TCP pose of the robot.
        Returns:
            (x, y, z, qw, qx, qy, qz), in flange coordinate
        """
        data = self.robot.get_ee_pose() # (position, quaternion(xyzw))
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return np.concatenate([pos, quat_wxyz]).tolist()

    def get_robot_state(self):
        # libfranka Gripper State has no element 'gripper_force'
        # libfranka Robot State has no element 'tcp_velocities'
        robot_state = self.robot.get_robot_state()
        # gripper_state = self.gripper.get_state()
        ee_pose = self.get_current_tcp()
        return {
            "leftRobotTCP": ee_pose, # (x, y, z, qw, qx, qy, qz), in flange coordinate 
            "leftRobotTCPWrench": robot_state.motor_torques_external.tolist(), # (fx, fy, fz, mx, my, mz)
            # "leftGripperState": [gripper_state.width, 0] # (width, force),libfranka Gripper State has no element 'gripper_force'
        }

    def go_home(self):
        home_joint_positions = [-0.2, -0.1, 1.5, -2.0, 0.2, 2.5, -0.4]
        homing_duration = 8.0
        logger.info(f"Moving Franka robot to home position: {home_joint_positions} with duration {homing_duration}s")
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(home_joint_positions),
            time_to_go=homing_duration
        )

    # TODO: Check whether the loop can run for 1000Hz ON THE DESKTOP
    def process_commands(self):
        """
        Main loop for processing commands and updating the interpolator.
        """
        if self.pose_interp is None:
            curr_flange_pose = self.get_ee_pose() # (x, y,z, rx, ry, rz)， in flange coordinate

            curr_time = time.monotonic()
            self.pose_interp = PoseTrajectoryInterpolator(
                times=[curr_time],
                poses=[curr_flange_pose]
            )
            self.last_waypoint_time = curr_time

        # start franka cartesian impedance policy
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(self.Kx),
            Kxd=torch.Tensor(self.Kxd)
        )

        t_start = time.monotonic()
        last_print = time.monotonic()
        count = 0
        iter_idx = 0

        while True:
            t_now = time.monotonic()
            flange_pos = self.pose_interp(t_now) # (x, y, z, rx, ry, rz), in flange coordinate

            self.robot.update_desired_ee_pose(
                position=torch.Tensor(flange_pos[:3]),
                orientation=torch.Tensor(st.Rotation.from_rotvec(flange_pos[3:]).as_quat()) # (qx, qy, qz, qw)
            )

            count += 1
            if t_now - last_print > 1.0:
                logger.info(f"update_desired_ee_pose called {count} times in last second")
                count = 0
                last_print = t_now

            '''
            Process high-level commands from VR
            command_queue: low-frequency moving command from VR
            target_time: timestamp where new target pose should be inserted into interpolator
            curr_time: the time base of interpolator
            last_waypoint_time: the last target pose in the interpolator
            '''                    
            try:
                command = self.command_queue.popleft()
                if command['cmd'] == Command.SCHEDULE_WAYPOINT.value:
                    target_pose = command['target_pose']
                    curr_time = t_now + self.control_cycle_time
                    target_time = float(command['target_time'])
                    # TODO: Determine whether we shoul calculate target time based on the current time here

                    self.pose_interp = self.pose_interp.schedule_waypoint(
                        pose=target_pose,
                        time=target_time,
                        curr_time=curr_time,
                        last_waypoint_time=self.last_waypoint_time
                    )
                    self.last_waypoint_time = target_time
            except IndexError:
                pass

            t_wait_util = t_start + (iter_idx + 1) * self.control_cycle_time
            precise_wait(t_wait_util, time_func=time.monotonic)
            iter_idx += 1

    def get_ee_pose(self):
        data = self.robot.get_ee_pose() # (position, quaternion(xyzw))
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()

    def run(self):
        self.go_home()
        logger.info("Franka moved to home position. Interpolation Controller started, waiting for commands...")
        command_thread = threading.Thread(target=self.process_commands, daemon=True)
        try:
            command_thread.start()
            logger.info("Start FastAPI Franka Server!")
            uvicorn.run(self.app, host=self.host_ip, port=self.port)
            command_thread.join()
        except Exception as e:
            command_thread.terminate()
            logger.exception(e)
        finally:
            command_thread.join()  
            self.robot.terminate_current_policy()
            logger.info("Franka Interpolation Controller terminated.")

def main():
    parser = argparse.ArgumentParser(description="Franka Server with Polymetis")
    parser.add_argument("--robot_ip", type=str, default='0.0.0.0', help="IP address of the robot")
    parser.add_argument("--host_ip", type=str, default="0.0.0.0", help="Host IP for FastAPI server")
    parser.add_argument("--port", type=int, default=8092, help="Port for FastAPI server")
    args = parser.parse_args()

    server = FrankaServer(
        robot_ip=args.robot_ip,
        host_ip=args.host_ip,
        port=args.port
    )
    server.run()

if __name__ == "__main__":
    main()