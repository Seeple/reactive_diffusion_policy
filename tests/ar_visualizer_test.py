# Test program: Send a hacked trajectory to the AR for visualization

import socket
import time
import math
import bson
from typing import List, Dict, Any


class TrajectoryPoint:
    """Data class for a single trajectory point."""
    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw
        }


class TrajectoryData:
    """Data class for trajectory data containing multiple points."""
    def __init__(self, points: List[TrajectoryPoint], timestamp: float):
        self.points = points
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'points': [point.to_dict() for point in self.points],
            'timestamp': self.timestamp
        }


class ARTrajectoryVisualizer:
    """
    A test program that generates and sends virtual trajectories to AR visualization.
    Sends 8 trajectory points per second to ChunkVisualizer.
    """
    
    def __init__(self, vr_server_ip: str = '127.0.0.1', vr_server_port: int = 10006):
        self.vr_server_ip = vr_server_ip
        self.vr_server_port = vr_server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"AR Trajectory Visualizer initialized. Sending to {vr_server_ip}:{vr_server_port}")
    
    def __init__(self, vr_server_ip: str = '127.0.0.1', vr_server_port: int = 10006):
        self.vr_server_ip = vr_server_ip
        self.vr_server_port = vr_server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Initialize the base trajectory (straight line)
        self.base_trajectory = self._create_base_trajectory()
        print(f"AR Trajectory Visualizer initialized. Sending to {vr_server_ip}:{vr_server_port}")
    
    def _create_base_trajectory(self, num_points: int = 8) -> List[TrajectoryPoint]:
        """
        Create the initial base trajectory as a straight line.
        
        Args:
            num_points: Number of points in the trajectory
            
        Returns:
            List of TrajectoryPoint objects forming a straight line
        """
        points = []
        
        # Create a straight line from (0, 0, 0.3) to (0, 0.4, 0.3)
        start_pos = [0.0, 0.0, 0.3]
        end_pos = [0.0, 0.4, 0.3]
        
        for i in range(num_points):
            # Linear interpolation along the line
            t = i / (num_points - 1) if num_points > 1 else 0
            
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            z = start_pos[2] + t * (end_pos[2] - start_pos[2])
            
            # Initial orientation (all zeros)
            roll = 0.0
            pitch = 0.0
            yaw = 0.0
            
            points.append(TrajectoryPoint(x, y, z, roll, pitch, yaw))
        
        return points
    
    def generate_evolving_trajectory(self, time_elapsed: float, num_points: int = 8) -> List[TrajectoryPoint]:
        """
        Generate an evolving trajectory based on the base trajectory with time-based variations.
        
        Args:
            time_elapsed: Time elapsed since start (in seconds)
            num_points: Number of points in the trajectory
            
        Returns:
            List of TrajectoryPoint objects with evolved positions and orientations
        """
        points = []
        
        # Variation parameters
        position_amplitude = 0.05  # Maximum position variation (5cm)
        orientation_amplitude = 0.3  # Maximum orientation variation (radians)
        
        for i in range(min(num_points, len(self.base_trajectory))):
            base_point = self.base_trajectory[i]
            
            # Create time-based variations for each point
            # Use different frequencies for each point to create wave-like motion
            point_freq = 0.5 + i * 0.1  # Different frequency for each point
            
            # Position variations
            x_variation = position_amplitude * math.sin(time_elapsed * point_freq)
            y_variation = position_amplitude * math.cos(time_elapsed * point_freq * 1.3)
            z_variation = position_amplitude * 0.5 * math.sin(time_elapsed * point_freq * 0.7)
            
            # Orientation variations
            roll_variation = orientation_amplitude * math.sin(time_elapsed * point_freq * 0.8)
            pitch_variation = orientation_amplitude * math.cos(time_elapsed * point_freq * 1.1)
            yaw_variation = orientation_amplitude * math.sin(time_elapsed * point_freq * 1.5)
            
            # Apply variations to base trajectory
            new_x = base_point.x + x_variation
            new_y = base_point.y + y_variation
            new_z = base_point.z + z_variation
            
            new_roll = base_point.roll + roll_variation
            new_pitch = base_point.pitch + pitch_variation
            new_yaw = base_point.yaw + yaw_variation
            
            points.append(TrajectoryPoint(new_x, new_y, new_z, new_roll, new_pitch, new_yaw))
        
        return points
    
    def send_trajectory(self, trajectory_points: List[TrajectoryPoint]):
        """
        Send trajectory data to AR visualization via UDP.
        
        Args:
            trajectory_points: List of trajectory points to send
        """
        current_time = time.time()
        trajectory_data = TrajectoryData(trajectory_points, current_time)
        
        # Convert to dictionary and then to BSON
        data_dict = trajectory_data.to_dict()
        packed_data = bson.dumps(data_dict)
        
        # Send via UDP
        try:
            self.socket.sendto(packed_data, (self.vr_server_ip, self.vr_server_port))
            print(f"Sent trajectory with {len(trajectory_points)} points at time {current_time:.3f}")
        except Exception as e:
            print(f"Error sending trajectory: {e}")
    
    def run_visualization_test(self, duration: int = 30):
        """
        Run the visualization test for specified duration.
        
        Args:
            duration: Test duration in seconds
        """
        print(f"Starting AR trajectory visualization test for {duration} seconds")
        print("Sending evolving trajectory with 8 points per second...")
        print("Initial trajectory: straight line from (0,0,0.3) to (0,0.4,0.3)")
        print("Each point will evolve with sinusoidal variations over time")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                loop_start = time.time()
                time_elapsed = time.time() - start_time
                
                # Generate evolving trajectory
                points = self.generate_evolving_trajectory(time_elapsed)
                
                # Send trajectory
                self.send_trajectory(points)
                
                frame_count += 1
                
                # Sleep to maintain 1 Hz (1 second interval)
                elapsed = time.time() - loop_start
                sleep_time = max(0, 1.0 - elapsed)
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        
        finally:
            total_time = time.time() - start_time
            print(f"\nTest completed. Sent {frame_count} trajectories in {total_time:.2f} seconds")
            print(f"Average rate: {frame_count/total_time:.2f} Hz")
    
    def close(self):
        """Close the socket connection."""
        self.socket.close()
        print("Socket connection closed")


def main():
    """Main function to run the AR trajectory visualization test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AR Trajectory Visualization Test')
    parser.add_argument('--ip', default='127.0.0.1', help='VR server IP address')
    parser.add_argument('--port', type=int, default=10006, help='VR server port')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    # Create and run the visualizer
    visualizer = ARTrajectoryVisualizer(args.ip, args.port)
    
    try:
        visualizer.run_visualization_test(args.duration)
    finally:
        visualizer.close()


if __name__ == '__main__':
    main()

