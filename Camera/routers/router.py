"""
This file contains all the routes for the camera microservice.
"""
import asyncio
from fastapi import APIRouter
from functools import lru_cache
from Camera.utils.upload_video import upload_video
from Camera.utils.camera import Camera
from Camera.utils.camera_reciver import CameraReceiver
from Camera.utils.video_recorder import VideoRecorder
from Camera.config.config import load_config

#-------------------------------------------------------------------#
# Configuration file path for the camera settings
CONFIG_PATH = "Camera/config/camera_config.yaml"

# Create an API router with a prefix for all camera-related routes
camera_router = APIRouter(prefix="/camera")

# Load config once for efficiency
config = load_config(CONFIG_PATH)
#-------------------------------------------------------------------#
@lru_cache()
def get_camera():
    """Cache camera instance to avoid repeated instantiation"""
    return Camera(CONFIG_PATH)

@lru_cache()
def get_camera_receiver():
    """Cache camera receiver instance to avoid repeated instantiation"""
    return CameraReceiver(CONFIG_PATH)

#-------------------------------------------------------------------#
@camera_router.get("/health")
async def health_check():
    return {"status": "OK"}

@camera_router.get("/status")
async def get_status():
    """Return which camera is available and its status."""
    try:
        camera = get_camera()
        if camera.is_available():
            print("[Camera Router] Hardware Camera Receiver")
            return {"camera_type": "Hardware Camera"}
        
        camera_receiver = get_camera_receiver()
        print("[Camera Router] Stream Camera Receiver")
        return {"camera_type": "Stream Camera Receiver"}
    except Exception as e:
        return {"error": str(e)}

@camera_router.get("/start")
async def start_camera():
    """Start the camera if available, otherwise start the camera receiver."""
    try:
        camera = get_camera()
        if camera.is_available():
            camera.start()
            return {"message": "Hardware camera started successfully."}
        
        camera_receiver = get_camera_receiver()
        camera_receiver.start()
        return {"message": "Camera receiver started successfully."}
    except Exception as e:
        return {"error": str(e)}
    
@camera_router.get("/stop")
async def stop_camera():
    """Stop the camera if available, otherwise stop the camera receiver."""
    try:
        camera = get_camera()
        if camera.is_available():
            camera.stop()
            return {"message": "Hardware camera stopped successfully."}
        
        camera_receiver = get_camera_receiver()
        camera_receiver.stop()
        return {"message": "Camera receiver stopped successfully."}
    except Exception as e:
        return {"error": str(e)}

@camera_router.get("/capture_frame")
async def capture_frame():
    try:
        camera = get_camera()
        if camera.is_available():
            color_frame, depth_frame = camera.capture_frame()
            if color_frame is None or depth_frame is None:
                raise ValueError("Captured frame is None.")
        else:
            camera_receiver = get_camera_receiver()
            color_frame, depth_frame = await camera_receiver.capture_frame()
            if color_frame is None or depth_frame is None:
                raise ValueError("Captured frame is None.")
        
        return {"color_frame": color_frame, "depth_frame": depth_frame}
    
    except ValueError as ve:
        print(f"[Camera Router] Frame capture error: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        print("[Camera Router] Unexpected error occurred")
        return {"error": "An unexpected error occurred. Check server logs."}

#-------------------------------------------------------------------#
@camera_router.get("/record")
async def record_video(num_recordings: int, action_name: str, objects: str):
    """
    Records a video and saves it using the VideoRecorder class.
    
    - `num_recordings`: Number of times to record
    - `action_name`: Name of the action for labeling
    - `objects`: Comma-separated list of objects to log
    """
    try:
        camera_receiver = get_camera_receiver()
        await camera_receiver.connect()

        object_list = objects.split(",") if objects else []
        recorder = VideoRecorder(camera_receiver, config, num_recordings, action_name, object_list)

        print(f"[Camera Router] Initiating video recording:")
        print(f"  - Number of recordings: {num_recordings}")
        print(f"  - Action: '{action_name}'")
        print(f"  - Objects: {object_list}")
        print("")

        await recorder.record_video()

        return {"message": f"Recording complete for action '{action_name}'", "num_recordings": num_recordings}
    
    except Exception as e:
        print(f"[Camera Router] ERROR during recording: {e}")
        return {"error": str(e)}

@camera_router.post("/upload_video_gcp")
async def upload_recorded_video(video_path: str):
    """
    Uploads a recorded video to Google Cloud Storage.
    - `video_path`: Path to the recorded video file.
    Returns:
        - A JSON response containing the uploaded video URL or an error message.
    """
    try:
        print(f"[Camera Router] Uploading video from path: {video_path}")
        video_url = upload_video(video_path)
        if video_url:
            return {"message": "Upload successful", "video_url": video_url}
        else:
            return {"error": "Video upload failed"}
    except Exception as e:
        print(f"[Camera Router] ERROR during upload: {e}")
        return {"error": str(e)}