'''
Franka interface client controller AND interpolator
A client controller for the Franka robot, which communicates with the Franka interface server, 
as well as interpolating the trajectory of the robot.
Expose the base controller through FASTAPI.
'''

import threading
import time
import numpy as np
import zerorpc
import torch
import enum
from fastapi import FastAPI, HTTPException
import scipy.spatial.transform as st
from loguru import logger
from typing import Dict
from collections import deque
from reactive_diffusion_policy.common.data_models import (TargetTCPRequest, MoveGripperRequest,         BimanualRobotStates)
import uvicorn
import argparse

from reactive_diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from reactive_diffusion_policy.common.precise_sleep import precise_sleep, precise_wait
from reactive_diffusion_policy.common.space_utils import mat_to_pose, pose_to_mat

tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_Gripper = 3

class FrankaInterpolationController:
    def __init__(self, 
                 robot_ip, 
                 robot_port, 
                 gripper_ip,
                 gripper_port,
                 host_ip='192.168.2.187', 
                 port=8092, 
                 Kx_scale=1.0,
                 Kxd_scale=1.0,
                 frequency=1000):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains.
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.host_ip = host_ip
        self.port = port
        self.frequency = frequency
        self.control_cycle_time = 1 / frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale

        self.robot = zerorpc.Client()
        self.robot.connect(f"tcp://{self.robot_ip}:{self.robot_port}")
        self.gripper = zerorpc.Client()
        self.gripper.connect(f"tcp://{gripper_ip}:{gripper_port}")

        self.app = FastAPI()
        self.setup_routes()

        self.command_queue = deque(maxlen=256)      # Command queue for low-frequency control signals

        self.pose_interp = None
        self.last_waypoint_time = None

    def setup_routes(self):
        @self.app.post('/clear_fault')
        async def clear_fault():
            """
            Clear any fault in the robot.
            """
            logger.warning("Fault occurred on franka robot server, trying to clear ...")
            self.robot.clear_fault()
            return {"message": "Fault cleared"}
        
        @self.app.get('/get_current_tcp/{robot_side}')
        async def get_current_tcp(robot_side: str):
            """
            Get the current TCP pose of the robot.
            """
            if robot_side != "left":
                raise HTTPException(status_code=400, detail="Only 'left' robot is supported")
            return self.robot.get_current_tcp()
        
        @self.app.post('/birobot_go_home')
        async def birobot_go_home():
            """
            Move the robot to its home position.
            """
            logger.info("Moving robot to home position")
            # TODO: check if there is determined homing positions and durations
            home_joint_positions = [0.0, -1.57, 0.0, -1.57, 0.0, 1.57, 0.0]
            homing_duration = 5.0
            self.robot.move_to_joint_positions(positions=home_joint_positions, 
                                               time_to_go=homing_duration)
            return {"message": "Robot moved to home position"}

        @self.app.get('/get_current_robot_state')
        async def get_current_robot_state() -> BimanualRobotStates:
            """
            Get the current state of the robot.
            """
            state = self.robot.get_robot_state()
            return BimanualRobotStates(**state)
        
        # Note: Franka gripper command is non-realtime and low-frequency
        # Thus don't need to interpolate the gripper command 
        @self.app.post('/move_gripper/{robot_side}')
        async def move_gripper(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            """
            Move the gripper to a target width with specified velocity and force limit.
            """
            if robot_side != "left":
                raise HTTPException(status_code=400, detail="Only 'left' robot is supported")
            self.gripper.move_gripper(request.width, request.velocity, request.force_limit)
            return {
                "message": f"{robot_side.capitalize()} gripper moving to width {request.width} "
                        f"with velocity {request.velocity} and force limit {request.force_limit}"
            }
         
        @self.app.post('/move_gripper_force/{robot_side}')
        async def move_gripper_force(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            """
            Close the gripper with a specified force limit.
            """
            if robot_side != "left":
                raise HTTPException(status_code=400, detail="Only 'left' robot is supported")
            self.gripper.move_gripper_force(request.force_limit)
            return {
                "message": f"{robot_side.capitalize()} gripper grasping with force limit {request.force_limit}"
            }
            
        @self.app.post('/stop_gripper/{robot_side}')
        async def stop_gripper(robot_side: str) -> Dict[str, str]:
            """
            Stop the gripper's current motion.
            """
            if robot_side != "left":
                raise HTTPException(status_code=400, detail="Only 'left' robot is supported")
            self.robot.stop_gripper()
            return {"message": f"{robot_side.capitalize()} gripper stopping"}
        
        @self.app.post('/move_tcp/{robot_side}')
        async def move_tcp(robot_side: str, request: TargetTCPRequest) -> Dict[str, str]:
            '''
            Move the robot to a target TCP pose.
            Add a new low-frequency target pose to the interpolator.
            '''
            if robot_side != "left":
                raise HTTPException(status_code=400, detail="Only 'left' robot is supported")
            
            target_pose = np.array(request.target_tcp)
            
            # TODO: replace the target time offset with actual frequency
            curr_time = time.monotonic()
            command_duration = 1/20
            target_time = curr_time + command_duration 
            
            self.command_queue.append({
                'cmd': Command.SCHEDULE_WAYPOINT.value,
                'target_pose': target_pose,
                'target_time': target_time
            })
            
            return {"message": f"Waypoint added for franka robot"}
            
        
    def process_commands(self):
        """
        Main loop for processing commands and updating the interpolator.
        """
        
        if self.pose_interp is None:
            curr_flange_pose = np.array(self.robot.get_ee_pose())
            curr_tip_pose = mat_to_pose(pose_to_mat(curr_flange_pose) @ tx_flange_tip)
            
            curr_time = time.monotonic()
            self.pose_interp = PoseTrajectoryInterpolator(
                times=[curr_time],
                poses=[curr_tip_pose]
            )
            self.last_waypoint_time = curr_time
            
        # start franka cartesian impedance policy
        self.robot.start_cartesian_impedance(
            Kx=self.Kx.tolist(),
            Kxd=self.Kxd.tolist()
        )
        
        t_start = time.monotonic()
        iter_idx = 0
                     
        while True:
            t_now = time.monotonic()
            
            # pose interpolation
            tip_pose = self.pose_interp(t_now)
            flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_flange)
            self.robot.update_desired_ee_pose(flange_pose.tolist())

            # Process high-level motion generation commands in the queue
            '''
            command_queue: low-frequency moving command from VR
            target_time: timestamp where new target pose should be inserted into interpolator
            curr_time: the time base of inetrpolator
            last_waypoint_time: the last target pose in the interpolator
            '''                          
            try:
                command = self.command_queue.popleft()
                logger.info(f"Processing command: {command}")
                if command['cmd'] == Command.SCHEDULE_WAYPOINT.value:
                    target_pose = command['target_pose']
                    curr_time = t_now + self.control_cycle_time # d_t means already implement an interpolated step
                    target_time = float(command['target_time'])
                    # command_duration = 1 / 20
                    # target_time = curr_time + command_duration

                    self.pose_interp = self.pose_interp.schedule_waypoint(
                        pose=target_pose,
                        time=target_time,
                        curr_time=curr_time,
                        last_waypoint_time=self.last_waypoint_time
                    )
                    self.last_waypoint_time = target_time
            except IndexError:
                pass  # No commands in the queue

            # # First loop successful, set ready_event
            # if iter_idx == 0:
            #     self.ready_event.set()

            # Regulate control frequency
            t_wait_util = t_start + (iter_idx + 1) * self.control_cycle_time
            precise_wait(t_wait_util, time_func=time.monotonic)
            iter_idx += 1

    def run(self):
        command_thread = threading.Thread(target=self.process_commands, daemon=True)
        try:
            command_thread.start()
            logger.info("Start FastAPI Franka Interpolation Controller!")
            uvicorn.run(self.app, host=self.host_ip, port=self.port)
            command_thread.join()
        except Exception as e:
            logger.exception(e)
        finally:
            self.robot.terminate_current_policy()
            logger.info("Franka Interpolation Controller terminated.")
        
def main():
    # TODO：Use config file to set up the parameters
    parser = argparse.ArgumentParser(description="Franka Interpolation Controller Debugging")
    parser.add_argument("--robot_ip", type=str, required=True, help="IP address of the robot")
    parser.add_argument("--robot_port", type=int, default=4242, help="Port of the robot server")
    parser.add_argument("--host_ip", type=str, default="0.0.0.0", help="Host IP for FastAPI server")
    parser.add_argument("--port", type=int, default=8092, help="Port for FastAPI server")
    args = parser.parse_args()

    controller = FrankaInterpolationController(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        host_ip=args.host_ip,
        port=args.port
    )
    controller.run()

if __name__ == "__main__":
    main()