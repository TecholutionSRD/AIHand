"""
This file contains all the routes for the camera microservice.
"""
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from Camera.functions.camera import Camera
from Camera.functions.camera_receiver import CameraReceiver
from Camera.functions.video_recorder import VideoRecorder
from Config.config import load_config

from Database.functions.upload_video import gcp_upload
from Database.functions.rlef import RLEFManager
from pydantic import BaseModel
from typing import List, Optional

#-------------------------------------------------------------------#
# Configuration file path for the camera settings
CONFIG_PATH = "camera_config.yaml"

# Create an API router with a prefix for all camera-related routes
camera_router = APIRouter(prefix="/camera")

# Load config once for efficiency
config = load_config(CONFIG_PATH)
#-------------------------------------------------------------------#
def get_camera():
    """Cache camera instance to avoid repeated instantiation"""
    return Camera(CONFIG_PATH)

def get_camera_receiver():
    """Cache camera receiver instance to avoid repeated instantiation"""
    return CameraReceiver(CONFIG_PATH)
#-------------------------------------------------------------------#
@camera_router.get("/health")
async def health_check():
    return {"status": "OK"}

@camera_router.get("/status")
async def get_status(camera: Camera = Depends(get_camera), camera_receiver: CameraReceiver = Depends(get_camera_receiver)):
    """Return which camera is available and its status."""
    try:
        camera = get_camera()
        if camera.is_available():
            print("[Camera Router] Hardware Camera Receiver")
            return {"camera_type": "Hardware Camera"}
        
        camera_receiver = get_camera_receiver()
        print("[Camera Router] Stream Camera Receiver")
        return {"camera_type": "Stream Camera"}
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
            frames = await camera_receiver.capture_frame()
            status, color_frame, depth_frame = frames.get('status','failed'), frames.get('color_frame'), frames.get('depth_frame')
            if color_frame is None or depth_frame is None:
                raise ValueError("Captured frame is None.")
        
        return {"status": status, "color_frame": color_frame, "depth_frame": depth_frame}
    
    except ValueError as ve:
        print(f"[Camera Router] Frame capture error: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        print("[Camera Router] Unexpected error occurred")
        return {"status":status, "color_frame": color_frame, "depth_frame": depth_frame}

#-------------------------------------------------------------------#
class RecordResponse(BaseModel):
    """Response model for video recording"""
    gcp_url: Optional[str] = None
    message: str

@camera_router.post("/record_video", response_model=RecordResponse)
async def record_video(action_name: str, objects: List[str]):
    """
    Records a video and saves it using the VideoRecorder class.

    - `action_name`: Name of the action for labeling
    - `objects`: List of objects associated with the action
    """
    if not action_name.strip():
        print("[Camera Router] Action name cannot be empty.")
        raise HTTPException(status_code=400, detail="Action name cannot be empty.")

    if not objects or not all(isinstance(obj, str) and obj.strip() for obj in objects):
        print("[Camera Router] Objects list must contain at least one valid string.")
        raise HTTPException(status_code=400, detail="Objects list must contain at least one valid string.")    

    try:
        num_recordings = 1
        camera_receiver = get_camera_receiver()
        connection = await camera_receiver.connect()
        if connection == False:
            print("[Camera Router] Failed to connect to camera receiver.")
            raise HTTPException(status_code=500, detail="Failed to connect to camera receiver.")

        recorder = VideoRecorder(camera_receiver, CONFIG_PATH, num_recordings, action_name, objects)

        print(f"[Camera Router] Initiating video recording...")
        print(f"  - Action: '{action_name}'")
        print(f"  - Objects: {objects}")

        video_path = await recorder.record_video()
        
        if not video_path:
            print("[Camera Router] Video recording failed, no file generated.")
            raise HTTPException(status_code=500, detail="Video recording failed.")

        print(f"[Camera Router] Video saved at {video_path}")

        # Upload to cloud storage (GCP & RLEF)
        try:
            print("[Camera Router] Uploading video to GCP & RLEF...")
            # gcp_url, rlef_result = await asyncio.gather(gcp_upload([video_path]), rlef_upload(action_name, video_path))
            gcp_url, rlef_result = None,None
            print(f"[Camera Router] Upload successful. GCP URL: {gcp_url}")
        except Exception as upload_error:
            print(f"[Camera Router] Upload failed: {upload_error}")
            gcp_url, rlef_result = None, None

        # Handle different upload cases
        if gcp_url and rlef_result:
            return RecordResponse(gcp_url=gcp_url, message="Video recorded and uploaded successfully to both GCP and RLEF.")
        elif gcp_url:
            return RecordResponse(gcp_url=gcp_url, message="Video recorded and uploaded to GCP only.")
        else:
            return RecordResponse(gcp_url=None, message="Video recorded but upload failed.")

    except HTTPException as http_error:
        print(f"[Camera Router] {http_error.detail}")
        raise http_error

    except Exception as unexpected_error:
        print(f"[Camera Router] {unexpected_error}")
        raise HTTPException(status_code=500, detail="Recording failed due to an internal error.")