"""
This file contains all the routes for the camera microservice.
"""
from fastapi import APIRouter
from functools import lru_cache
from utils.camera import Camera
from utils.camera_reciver import CameraReceiver
#-------------------------------------------------------------------#
CONFIG_PATH = "config/camera_config.yaml"
camera_router = APIRouter(prefix="/camera")
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
@camera_router.get("/status")
async def get_status():
    """Return which camera is available and its status."""
    try:
        camera = get_camera()
        if camera.is_available():
            return {
                "camera_type": "Hardware Camera",
            }
        
        camera_receiver = get_camera_receiver()
        return {
            "camera_type": "Camera Stream Receiver",
        }
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
    """Capture a frame from the camera if available, otherwise capture a frame from the camera receiver."""
    try:
        camera = get_camera()
        if camera.is_available():
            color_frame,depth_frame = camera.capture_frame()
            if color_frame is None or depth_frame is None:
                return {"error": "No frame available."}
            return {"color_frame": color_frame, "depth_frame": depth_frame}
        
        camera_receiver = get_camera_receiver()
        color_frame,depth_frame = await camera_receiver.capture_frame()
        if color_frame is None or depth_frame is None:
            return {"error": "No frame available."}
        return {"color_frame": color_frame, "depth_frame": depth_frame}
    except Exception as e:
        return {"error": str(e)}
#-------------------------------------------------------------------#