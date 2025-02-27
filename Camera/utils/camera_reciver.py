"""
This file contains the code for receving images(frames) from the MQTT server.
"""
import uuid
import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
import io
import os
import sys
import pyrealsense2 as rs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config.config import load_config

class CameraReceiver:
    """
    Class to receive and process image frames from an MQTT WebSocket server.
    """
    def __init__(self, config_path:str):
        """
        Initializes the CameraReceiver with the provided configuration file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        config = load_config(config_path)
        self.config = config.get('Stream', {})
        self.camera_config = config.get('Camera', {})
        self.websocket_server = self.config.get("Websocket_server", "")
        self.websocket_topic = self.config.get("Websocket_topic", "")
        self.save_path = self.config.get("save_path", "data/captured_frames")
        self.websocket = None
        self.running = False
        self.display_task = None
    
    async def connect(self, retries:int=3, delay:int=2) -> bool:
        """
        Establishes a connection to the WebSocket server with retries.
        
        Args: 
            retries (int): Number of retry attempts.
            delay (int): Delay in seconds between retries.
        Returns:
            bool: True or False output.
        """
        uri = f"{self.websocket_server}{self.websocket_topic}"
        for attempt in range(retries):
            try:
                self.websocket = await websockets.connect(uri)
                print(f"[Camera] Connected to WebSocket Server at {uri}")
                return True
            except (websockets.exceptions.WebSocketException, ConnectionRefusedError) as e:
                print(f"[Camera] Connection attempt {attempt+1} failed: {e}")
                await asyncio.sleep(delay*(2**attempt))
        print("[Camera] Failed to connect after multiple attempts")
        self.websocket = None
        return False

    def _get_intrinsics(self, location:str="India", camera_name:str="D435I"):
        """
        Get the camera intrinsics from the configuration file.
        """
        intrinsics = rs.intrinsics()
        color_intrinsics = self.camera_config[camera_name][location]['Intrinsics']['Color_Intrinsics']
        intrinsics.width = 640
        intrinsics.height = 480
        intrinsics.ppx = color_intrinsics.get('ppx', 0)
        intrinsics.ppy = color_intrinsics.get('ppy', 0) 
        intrinsics.fx = color_intrinsics.get('fx', 0)
        intrinsics.fy = color_intrinsics.get('fy', 0)
        intrinsics.model = rs.distortion.inverse_brown_conrady
        intrinsics.coeffs = [0, 0, 0, 0, 0]
        return intrinsics

    async def decode_frames(self):
        """
        Receives and decodes both color and depth frames from JSON data.
        
        Returns:
            tuple: (color_frame, depth_frame) if successful, else (None, None)
        """
        if self.websocket is None:
            print("[Camera] WebSocket connection is not established.")
            return None, None
        
        try:
            json_data = await self.websocket.recv()
            frame_data = json.loads(json_data)
            
            # Decode color frame
            color_data = base64.b64decode(frame_data.get('color', ""))
            color_arr = np.frombuffer(color_data, np.uint8)
            color_frame = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
            
            # Decode depth frame
            depth_data = base64.b64decode(frame_data.get('depth', ""))
            depth_bytes = io.BytesIO(depth_data)
            depth_frame = np.load(depth_bytes, allow_pickle=True)
            
            return color_frame, depth_frame
        except (json.JSONDecodeError, KeyError, ValueError, cv2.error) as e:
            print(f"[Camera] Decoding error: {e}")
        except Exception as e:
            print(f"[Camera] Unexpected error: {e}", exc_info=True)
            raise
    
    async def frames(self):
        """
        Asynchronous generator that continuously receives frames.
        
        Yields:
            tuple: (color_frame, depth_frame) until stopped.
        """
        if self.websocket is None:
            print("[Camera] WebSocket connection is missing. Exiting frame loop.")
            return

        try:
            while self.running:
                color_frame, depth_frame = await self.decode_frames()
                yield color_frame, depth_frame
        except asyncio.CancelledError:
            print("[Camera] Frame receiving loop has been cancelled.")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[Camera] WebSocket connection closed unexpectedly: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """
        Closes the WebSocket connection and releases resources.
        """
        if self.websocket:
            await self.websocket.close()
            print(f"[Camera] WebSocket connection closed.")
            self.websocket = None
        self.running = False

    async def display_async(self):
        """
        Connects to the WebSocket server and displays the received frames.
        """
        connection_success = await self.connect()
        
        if not connection_success:
            print(f"[Camera] Failed to connect to WebSocket server.")
            return {"error": "Failed to connect to WebSocket server."}
        
        if self.websocket:
            async for color_frame, depth_frame in self.frames():
                if color_frame is not None:
                    cv2.imshow("Color Frame", color_frame)
                if depth_frame is not None:
                    normalized_depth = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                    normalized_depth = normalized_depth.astype(np.uint8)
                    cv2.imshow("Depth Frame", normalized_depth)
                if (cv2.waitKey(1) & 0xFF == ord('q')) or not self.running:
                    break
        
        cv2.destroyAllWindows()
        await self.cleanup()
        return {"message": "Display stopped."}

    async def stop_display_async(self):
        """
        Stops the display by setting running flag to False and cleaning up.
        """
        self.running = False
        cv2.destroyAllWindows()
        await self.cleanup()
        return {"message": "Display stopped asynchronously."}

    async def capture_frame(self):
        """
        Connects to the WebSocket server, receives one frame, and saves it to the specified location.
        
        Returns:
            dict: A dictionary containing paths to the 'rgb' and 'depth' directories.
        """
        connection_success = await self.connect()
        
        if not connection_success:
            return {"error": "Failed to connect to WebSocket server."}
        id = uuid.uuid4()
        rgb_dir = f"{self.save_path}/{id}/rgb"
        depth_dir = f"{self.save_path}/{id}/depth"
        
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        print(f"[Camera] Using directories at {self.save_path}")

        color_frame_path = None
        depth_frame_path = None

        if self.websocket:
            self.running = True
            async for color_frame, depth_frame in self.frames():
                if color_frame is None or depth_frame is None:
                    await self.cleanup()
                    return {"error": "Failed to capture frames: Color or Depth frame is None"}

                try:
                    color_frame_path = f"{rgb_dir}/image_0.jpg"
                    cv2.imwrite(color_frame_path, color_frame)
                    print(f"[Camera] Saved color frame to {color_frame_path}")

                    depth_frame_path = f"{depth_dir}/image_0.npy"
                    np.save(depth_frame_path, depth_frame)
                    print(f"[Camera] Saved depth frame to {depth_frame_path}")

                    self.running = False
                    break
                except Exception as e:
                    await self.cleanup()
                    return {"error": f"Failed to save frames: {str(e)}"}

        await self.cleanup()

        return {
            "rgb": color_frame_path,
            "depth": depth_frame_path
        }

    async def start_display(self):
        """
        Asynchronously starts the display process.
        This method sets the running flag to True and then calls the display_async method
        to handle the display functionality.
        Returns:
            None
        """
        self.running = True
        await self.display_async()


    def start(self, mode="display"):
        """
        Starts the camera receiver in the specified mode.
        Args:
            mode (str): The mode in which to start the camera receiver. 
                        Options are "display" to start displaying the camera feed, 
                        or "capture" to capture a single frame. Default is "display".
        Returns:
            If mode is "capture", returns the captured frame.
            If mode is "display", runs the display loop until stopped.
        """
        self.running = True
        if mode == "display":
            asyncio.run(self.start_display())
        elif mode == "capture":
            return asyncio.run(self.capture_frame())
    
    async def stop_display_async(self):
        """
        Asynchronously stops the display and closes the websocket connection.
        This method sets the running flag to False, closes the websocket connection
        if it exists, and destroys all OpenCV windows.
        Returns:
            None
        """
        self.running = False
        if self.websocket:
            await self.websocket.close()
        cv2.destroyAllWindows()

    def stop(self):
        """
        Stops the camera receiver by setting the running flag to False.
        
        Returns:
            dict: Status message
        """
        if not self.running:
            return {"message": "Camera receiver is not running."}
            
        self.running = False
        if hasattr(self, 'display_thread') and self.display_thread.is_alive():
            self.display_thread.join(timeout=5.0)
            
        cv2.destroyAllWindows()
        return {"message": "Camera receiver stopped successfully."}
 