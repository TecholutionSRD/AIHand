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
import logging
import os
import sys
import pyrealsense2 as rs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config.config import load_config

class CameraReceiver:
    """
    Class to receive and process image frames from an MQTT WebSocket server.
    """
    def __init__(self, config_path:str):
        """
        Initializes the CameraReceiver with the provided configuration.
        
        Args:
            config (path): Configuration dictionary containing WebSocket settings.
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
    
    async def connect(self):
        """
        Establishes a connection to the WebSocket server.
        """
        try:
            uri = f"{self.websocket_server}{self.websocket_topic}"
            self.websocket = await websockets.connect(uri)
            logging.info(f"Connected to WebSocket server at {uri}")
            return True
        except (websockets.exceptions.InvalidURI, websockets.exceptions.InvalidHandshake, ConnectionRefusedError) as e:
            logging.error(f"Error connecting to WebSocket server: {e}")
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
            logging.warning("WebSocket connection is not established.")
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
        except (json.JSONDecodeError, KeyError, ValueError, cv2.error, Exception) as e:
            logging.error(f"Error decoding frames: {e}")
            return None, None
    
    async def frames(self):
        """
        Asynchronous generator that continuously receives frames.
        
        Yields:
            tuple: (color_frame, depth_frame) until stopped.
        """
        if self.websocket is None:
            logging.error("WebSocket connection is missing. Exiting frame loop.")
            return

        try:
            while self.running:
                color_frame, depth_frame = await self.decode_frames()
                yield color_frame, depth_frame
        except asyncio.CancelledError:
            logging.info("Frame receiving loop has been cancelled.")
        except websockets.exceptions.ConnectionClosed as e:
            logging.warning(f"WebSocket connection closed unexpectedly: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """
        Closes the WebSocket connection and releases resources.
        """
        if self.websocket:
            await self.websocket.close()
            logging.info("WebSocket connection closed.")
            self.websocket = None
        self.running = False

    async def display_async(self):
        """
        Connects to the WebSocket server and displays the received frames.
        """
        connection_success = await self.connect()
        
        if not connection_success:
            logging.error("Failed to connect to WebSocket server.")
            return {"error": "Failed to connect to WebSocket server."}
        
        if self.websocket:
            async for color_frame, depth_frame in self.frames():
                if color_frame is not None:
                    cv2.imshow("Color Frame", color_frame)
                if depth_frame is not None:
                    # Normalize depth values for better visualization
                    normalized_depth = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                    normalized_depth = normalized_depth.astype(np.uint8)
                    cv2.imshow("Depth Frame", normalized_depth)
                
                # Check if we should stop
                if (cv2.waitKey(1) & 0xFF == ord('q')) or not self.running:
                    break
        
        cv2.destroyAllWindows()
        await self.cleanup()
        return {"message": "Display stopped."}

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
        
        # Fixed the directory creation logic
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        logging.info(f"Using directories at {self.save_path}")

        color_frame_path = None
        depth_frame_path = None

        if self.websocket:
            # Set running to True just for a single frame capture
            self.running = True
            async for color_frame, depth_frame in self.frames():
                if color_frame is not None:
                    color_frame_path = f"{rgb_dir}/image_0.jpg"
                    cv2.imwrite(color_frame_path, color_frame)
                    logging.info(f"Saved color frame to {color_frame_path}")
                
                if depth_frame is not None:
                    depth_frame_path = f"{depth_dir}/image_0.npy"
                    np.save(depth_frame_path, depth_frame)
                    logging.info(f"Saved depth frame to {depth_frame_path}")
                
                # Only capture one frame
                self.running = False
                break

        await self.cleanup()
        
        return {
            "rgb": color_frame_path,
            "depth": depth_frame_path
        }

    def start(self, mode="display"):
        """
        Start the camera receiver in the specified mode.
        
        Args:
            mode (str): Operating mode - either "display" or "capture"
        
        Returns:
            dict: Status information or captured frame paths
        """
        self.running = True
        
        try:
            if mode == "display":
                # Create a new thread that runs its own event loop for display
                def run_display_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self.display_async())
                    finally:
                        loop.close()
                
                import threading
                self.display_thread = threading.Thread(target=run_display_in_thread)
                self.display_thread.daemon = True
                self.display_thread.start()
                return {"message": "Camera receiver started in display mode. Press 'q' to quit."}
                
            elif mode == "capture":
                # For capture mode, run in a separate event loop
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(self.capture_frame_async())
                finally:
                    loop.close()
            else:
                self.running = False
                return {"error": f"Unknown mode: {mode}. Use 'display' or 'capture'."}
                
        except Exception as e:
            self.running = False
            logging.error(f"Error starting camera receiver: {e}")
            return {"error": f"Error starting camera receiver: {e}"}
    
    def stop(self):
        """
        Stops the camera receiver by setting the running flag to False.
        
        Returns:
            dict: Status message
        """
        if not self.running:
            return {"message": "Camera receiver is not running."}
            
        self.running = False
        
        # Wait for the display thread to finish if it exists
        if hasattr(self, 'display_thread') and self.display_thread.is_alive():
            self.display_thread.join(timeout=5.0)
            
        cv2.destroyAllWindows()
        return {"message": "Camera receiver stopped successfully."}
    
if __name__ == "__main__":
    import argparse
    import time

    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Camera Receiver for MQTT WebSocket stream")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["display", "capture"], default="display",
                        help="Operation mode: 'display' for live view, 'capture' for saving frames")
    parser.add_argument("--duration", type=int, default=0,
                        help="Duration in seconds to run in display mode (0 for indefinite)")
    args = parser.parse_args()
    
    try:
        # Initialize the camera receiver
        receiver = CameraReceiver(args.config)
        
        if args.mode == "display":
            print(f"Starting camera receiver in display mode. Press 'q' to quit.")
            result = receiver.start(mode="display")
            print(result["message"])
            
            # If duration is specified, wait and then stop
            if args.duration > 0:
                print(f"Will run for {args.duration} seconds...")
                time.sleep(args.duration)
                stop_result = receiver.stop()
                print(stop_result["message"])
            else:
                # Keep the main thread alive
                try:
                    while receiver.running:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("Keyboard interrupt received. Stopping...")
                    receiver.stop()
        
        elif args.mode == "capture":
            print("Capturing a single frame...")
            result = receiver.start(mode="capture")
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Captured frames saved to:")
                print(f"  RGB: {result['rgb']}")
                print(f"  Depth: {result['depth']}")
                
    except Exception as e:
        print(f"Error: {e}")
        # Make sure to stop the receiver if an exception occurs
        if 'receiver' in locals():
            receiver.stop()