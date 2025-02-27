from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from functools import lru_cache
import asyncio
from VisionAI.utils.vision_detection import GeminiInference
from VisionAI.config.config import load_config
from Camera.utils.camera_reciver import CameraReceiver
#-------------------------------------------------------------------#
# Configuration file path for the camera settings
CONFIG_PATH = "VisionAI/config/vision_ai_config.yaml"

# Create an API router with a prefix for all vision-related routes
visionai_router = APIRouter(prefix="/vision_ai")

# Load config once for efficiency
config = load_config(CONFIG_PATH)
#-------------------------------------------------------------------#
@lru_cache()
def get_gemini():
    """Cache Gemini Model instance to avoid repeated loading."""
    return GeminiInference(config)

@lru_cache()
def get_camera_receiver():
    """Cache CameraReceiver instance to avoid repeated instantiation."""
    return CameraReceiver(CONFIG_PATH)

#-------------------------------------------------------------------#
# Health Check Endpoint
@visionai_router.get("/health")
async def health_check():
    """Check the health of the VisionAI microservice."""
    return {"status": "OK"}

#-------------------------------------------------------------------#
# Object Detection
class ObjectDetectionResponse(BaseModel):
    """
    Response model for object detection API.
    
    Attributes:
        objects (List[str]): A list of detected object names.
    """
    objects: List[str]

@visionai_router.get("/detect", response_model=ObjectDetectionResponse)
async def detect():
    """
    Perform object detection using the Gemini inference model.
    
    Returns:
        ObjectDetectionResponse: List of detected object names.
    """
    try:
        camera = get_camera_receiver()
        gemini = get_gemini()
        objects = await gemini.detect_all(camera)
        return ObjectDetectionResponse(objects=objects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-------------------------------------------------------------------#
# Object Center Detection
class ObjectCenterResponse(BaseModel):
    """
    Response model for object center detection API.
    
    Attributes:
        center (Optional[Tuple[int, int]]): Center coordinates of the detected object.
        box (Optional[List[int]]): Bounding box coordinates.
        confidence (Optional[float]): Confidence score.
    """
    center: Optional[List[int]]
    box: Optional[List[int]]
    confidence: Optional[float]


@visionai_router.get("/detect_object_center", response_model=ObjectCenterResponse)
async def detect_object_center(target_class: str):
    """
    Detect the center of a specified object.
    
    Args:
        target_class (str): The class of object to detect.
    
    Returns:
        ObjectCenterResponse: Center, bounding box, and confidence score.
    """
    try:
        camera = get_camera_receiver()
        gemini = get_gemini()
        result = await gemini.get_object_center(await camera.capture_frame(), target_class)
        if not result:
            raise HTTPException(status_code=404, detail="Object not found")
        return ObjectCenterResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-------------------------------------------------------------------#
# Set Target Classes Request Model
class TargetClassesRequest(BaseModel):
    """
    Request model for setting target detection classes.
    
    Attributes:
        target_classes (List[str]): List of target object classes.
    """
    target_classes: List[str]

@visionai_router.post("/set_target_classes")
async def set_target_classes(request: TargetClassesRequest):
    """
    Update the target classes for object detection.
    
    Args:
        request (TargetClassesRequest): Target classes list.
    
    Returns:
        dict: Confirmation message.
    """
    try:
        gemini = get_gemini()
        gemini.set_target_classes(request.target_classes)
        return {"message": "Target classes updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-------------------------------------------------------------------#
# Set Inference State
class InferenceStateRequest(BaseModel):
    """
    Request model for setting inference mode.
    
    Attributes:
        state (bool): New inference mode state.
    """
    state: bool


@visionai_router.post("/set_inference_state")
async def set_inference_state(request: InferenceStateRequest):
    """
    Enable or disable inference mode.
    
    Args:
        request (InferenceStateRequest): New inference state.
    
    Returns:
        dict: Confirmation message.
    """
    try:
        gemini = get_gemini()
        gemini.set_inference_state(request.state)
        return {"message": "Inference state updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-------------------------------------------------------------------#
# Set Capture State
class CaptureStateRequest(BaseModel):
    """
    Request model for setting capture mode.
    
    Attributes:
        state (bool): New capture mode state.
    """
    state: bool

@visionai_router.post("/set_capture_state")
async def set_capture_state(request: CaptureStateRequest):
    """
    Enable or disable capture mode.
    
    Args:
        request (CaptureStateRequest): New capture state.
    
    Returns:
        dict: Confirmation message.
    """
    try:
        gemini = get_gemini()
        gemini.set_capture_state(request.state)
        return {"message": "Capture state updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
