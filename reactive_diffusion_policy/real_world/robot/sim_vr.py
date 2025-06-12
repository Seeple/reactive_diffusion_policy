import numpy as np
import time
import requests
from loguru import logger
import threading
from reactive_diffusion_policy.common.data_models import UnityMes, HandMes
import scipy.spatial.transform as st

class VirtualVRDevice:
    def __init__(self,
                teleop_server_ip: str = "192.168.2.187",
                teleop_server_port: int = 8082,
                frequency: float = 90.0,
                radius: float = 0.2,  
                height: float = 0.5, 
                angular_velocity: float = 1.0):
        """
        Args:
            teleop_server_ip: 远程操作服务器 IP
            teleop_server_port: 远程操作服务器端口
            frequency: 发送频率 (Hz)
            radius: 圆形轨迹半径 (m)
            height: 圆形轨迹高度 (m)
            angular_velocity: 角速度 (rad/s)
        """
        self.teleop_server_ip = teleop_server_ip
        self.teleop_server_port = teleop_server_port
        self.period = 1.0 / frequency
        self.radius = radius
        self.height = height
        self.angular_velocity = angular_velocity
        
        self.session = requests.Session()
        
        self.running = False
        self.start_time = None
        
    def _generate_circle_pose(self, t: float):
        angle = self.angular_velocity * t
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        z = self.height
        
        rotation = st.Rotation.from_euler('xyz', [0, 0, angle])
        quat = rotation.as_quat()  # [qx, qy, qz, qw]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        
        return np.array([x, y, z]), quat
    
    def _create_unity_message(self, t: float) -> UnityMes:
        pos, quat = self._generate_circle_pose(t)
        
        left_hand = HandMes(
            wristPos=pos.tolist(),
            wristQuat=quat.tolist(),
            triggerState=0.0,  
            buttonState=[False, False, False, False, True] 
        )
        
        right_hand = HandMes(
            wristPos=[0.3, 0.0, self.height],
            wristQuat=[1.0, 0.0, 0.0, 0.0],
            triggerState=0.0,
            buttonState=[False, False, False, False, False]
        )
        
        return UnityMes(
            timestamp=time.monotonic(),
            leftHand=left_hand,
            rightHand=right_hand
        )
    
    def start(self):
        self.running = True
        self.start_time = time.monotonic()
        self.send_thread = threading.Thread(target=self._send_loop)
        self.send_thread.start()
        logger.info(f"Virtual VR device started, sending to {self.teleop_server_ip}:{self.teleop_server_port}")
        
    def stop(self):
        self.running = False
        if hasattr(self, 'send_thread'):
            self.send_thread.join()
        logger.info("Virtual VR device stopped")
        
    def _send_loop(self):
        while self.running:
            loop_start = time.monotonic()
            
            try:
                t = time.monotonic() - self.start_time
                message = self._create_unity_message(t)
                
                url = f"http://{self.teleop_server_ip}:{self.teleop_server_port}/unity"
                response = self.session.post(url, json=message.model_dump())
                
                if response.status_code != 200:
                    logger.warning(f"Failed to send message: {response.status_code}")
                
            except Exception as e:
                logger.error(f"Error in send loop: {e}")
            
            elapsed = time.monotonic() - loop_start
            if elapsed < self.period:
                time.sleep(self.period - elapsed)

if __name__ == "__main__":
    device = VirtualVRDevice(
        teleop_server_ip="192.168.2.187",
        teleop_server_port=8082,
        frequency=90.0,
        radius=0.2,
        height=0.5,
        angular_velocity=1.0
    )
    
    try:
        device.start()

        time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Stopping virtual VR device...")
    finally:
        device.stop()