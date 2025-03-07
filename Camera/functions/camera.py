"""
This module provides functionality to interface with an Intel RealSense camera.
It allows capturing images directly from the camera when an MQTT server is not available.

Dependencies:
- pyrealsense2
- numpy
- OpenCV
- A configuration module for loading settings

Usage:
    Run this script directly to start the camera and display the feed.
    Press 'q' to exit the video stream.
"""
import logging
import uuid
import os
import sys
import pyrealsense2 as rs
import numpy as np
import cv2
from Config.config import load_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
 
class Camera:
    """
    A class to manage an Intel RealSense D435i camera.
    
    This class provides methods to initialize, start, capture frames, and stop the camera.
    """
    
    def __init__(self, config_path:str):
        """
        Initializes the RealSense camera with settings from the configuration file.
        
        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config_data = load_config(config_path)
        self.save_path = self.config_data.get("Stream", {}).get("Save_path", None)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        camera_settings = self.config_data["Camera"]["D435I"]["India"]["Intrinsics"]["Color_Intrinsics"]
        self.width = camera_settings["width"]
        self.height = camera_settings["height"]
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        
        print("[Camera] RealSense Camera Initialized")

    def start(self):
        """
        Starts the RealSense camera pipeline, enabling data streaming.
        """
        self.pipeline.start(self.config)
        print("[Camera] RealSense Camera Started")

    def capture_frame(self, retries: int = 3, timeout: int = 5):
        """
        Captures a single frame from the RealSense camera with retries and timeout.

        Returns:
            tuple: (color_image, depth_image) as numpy arrays.
                If self.save_path is set, returns a dict with paths to saved files.
        """
        attempt = 0
        color_image, depth_image = None, None

        while attempt < retries:
            attempt += 1
            print(f"[Camera] Attempt {attempt}/{retries} to capture frame...")

            try:
                start_time = time.time()

                # Attempt to capture frame with timeout
                while time.time() - start_time < timeout:
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()

                    if color_frame and depth_frame:
                        color_image = np.asanyarray(color_frame.get_data())
                        depth_image = np.asanyarray(depth_frame.get_data())
                        break  # Success, exit loop

                if color_image is None or depth_image is None:
                    raise ValueError("Captured frame is None.")

                print(f"[Camera] Frame captured successfully on attempt {attempt}")

                # Save the captured frames if save_path is set
                if self.save_path:
                    id = uuid.uuid4()
                    rgb_dir = f"{self.save_path}/{id}/rgb"
                    depth_dir = f"{self.save_path}/{id}/depth"

                    os.makedirs(rgb_dir, exist_ok=True)
                    os.makedirs(depth_dir, exist_ok=True)

                    color_frame_path = f"{rgb_dir}/image_0.jpg"
                    depth_frame_path = f"{depth_dir}/image_0.npy"

                    cv2.imwrite(color_frame_path, color_image)
                    np.save(depth_frame_path, depth_image)

                    print(f"[Camera] Saved frames: {color_frame_path}, {depth_frame_path}")
                    return color_image, depth_image, {"rgb": color_frame_path, "depth": depth_frame_path}

                return color_image, depth_image

            except ValueError as ve:
                print(f"[Camera] Frame error: {ve}")
            except Exception as e:
                print(f"[Camera] Unexpected error: {e}")

            if attempt < retries:
                print("[Camera] Retrying in 0.5 seconds...")
                time.sleep(0.5)

        print("[Camera] Failed to capture frame after multiple attempts.")
        return None, None, {"error": "Failed to capture frame after multiple attempts."}
    
    def stop(self):
        """
        Stops the RealSense camera pipeline, terminating data streaming.
        """
        self.pipeline.stop()
        print("[Camera] RealSense Camera Stopped")

    def is_available(self):
        """
        Checks if the Intel RealSense camera is connected.
        
        Returns:
            bool: True if the camera is connected, False otherwise.
        """
        try:
            ctx = rs.context()
            if len(ctx.devices) > 0:
                print("[Camera] RealSense Camera is connected")
                return True
            else:
                print("[Camera] No RealSense Camera connected")
                return False
        except Exception as e:
            print(f"[Camera] Error checking camera connection: {e}")
            return False

