from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from functools import lru_cache
from PIL import Image
import asyncio
import numpy as np
from VisionAI.functions.llm_router import query_router
from VisionAI.functions.vision_detection import GeminiInference
from Config.config import load_config
from Camera.functions.camera_receiver import CameraReceiver
#-------------------------------------------------------------------#
CONFIG_PATH = "vision_ai_config.yaml"

visionai_router = APIRouter(prefix="/vision")

config = load_config(CONFIG_PATH)
#-------------------------------------------------------------------#
@lru_cache()
def get_gemini():
    """Cache Gemini Model instance to avoid repeated loading."""
    return GeminiInference(CONFIG_PATH)

@lru_cache()
def get_camera_receiver():
    """Cache CameraReceiver instance to avoid repeated instantiation."""
    return CameraReceiver(CONFIG_PATH)
#-------------------------------------------------------------------#
# Health Check Endpoint
@visionai_router.get("/health")
async def health_check():
    """Check the health of the VisionAI microservice."""
    try:
        camera = get_camera_receiver()
        gemini = get_gemini()
        if camera and gemini:
            return {"status": "OK"}
        else:
            return {"status": "Not OK"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-------------------------------------------------------------------#
# Object Detection
class ObjectDetectionResponse(BaseModel):
    """
    Response model for object detection API.
    
    Attributes:
        objects (List[str]): List of detected object names.
        status (str): Status of the detection process.
        message (str): Additional information or error message.
        confidence_scores (Dict[str, float]): Confidence scores for each detected object.
    """
    objects: Optional[List[str]] = None
    status: str
    message: Optional[str] = None

@visionai_router.get("/detect", response_model=ObjectDetectionResponse)
async def detect():
    """
    Perform object detection using the Gemini inference model.
    Returns:
        ObjectDetectionResponse: Detection results including objects, status, and confidence scores.
    Raises:
        HTTPException: If camera capture fails or detection encounters an error.
    """
    try:
        camera = get_camera_receiver()
        gemini = get_gemini()

        connection = await camera.connect()
        if not connection:
            print("[Vision Router] Failed to connect to camera receiver.")
            raise HTTPException(status_code=500, detail="Failed to connect to camera receiver.")

        frame_data = await camera.capture_frame()
        
        if frame_data["status"] == "failed":
            raise HTTPException(status_code=500, detail="Camera failed to capture a valid frame.")

        color_frame_path = frame_data.get("color_frame")
        if not color_frame_path or color_frame_path == "None":
            raise HTTPException(status_code=500, detail="No valid image path returned by camera.")
        try:
            color_frame = Image.open(color_frame_path)
        except Exception as e:
            print(f"[Vision Router] Failed to process image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

        detection_result = gemini.detect(color_frame)

        if not detection_result:
            print("[Vision Router] No objects detected")
            return ObjectDetectionResponse(status="success", message="No objects detected", objects=[])

        return ObjectDetectionResponse(status="success", objects=detection_result, message="Objects detected successfully")

    except Exception as e:
        print(f"[Vision Router] Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
#-------------------------------------------------------------------------------------------------#
# Object Center Detection
class ObjectCenterResponse(BaseModel):
    """
    Response model for object center detection API.

    Attributes:
        object_name (str): The name of the detected object.
        center_coordinates (List[float]): Real-world center coordinates [x, y, z].
    """
    object_name: str
    center_coordinates: List[float]


@visionai_router.post("/detect_object_centers", response_model=List[ObjectCenterResponse])
async def detect_object_centers(target_classes: List[str]):
    """
    Detect the real-world center of multiple specified objects.

    Args:
        target_classes (List[str]): The list of object classes to detect.

    Returns:
        List[ObjectCenterResponse]: A list of detected objects with centers, bounding boxes, and confidence scores.
    """
    try:
        print("[Vision Router] Starting multiple object center detection")

        camera = get_camera_receiver()
        gemini = get_gemini()
        intrinsics = camera._get_intrinsics()

        # frames = await camera.capture_frame()
        # if not frames or frames.get("status") != "success":
        #     print("[Vision Router] Failed to capture frames from camera")
        #     raise HTTPException(status_code=500, detail="Failed to capture frames from camera.")

        # color_frame = frames.get("color_frame")
        # depth_frame = frames.get("depth")

        color_frame_path = "C:/Users/ASUS/Desktop/Techolution/AIHand/data/captured_frames/1ede1de0-cb15-4460-b55b-69cf138e7e07/rgb/image_0.jpg"
        depth_frame_path = "C:/Users/ASUS/Desktop/Techolution/AIHand/data/captured_frames/1ede1de0-cb15-4460-b55b-69cf138e7e07/depth/image_0.npy"
        color_frame = Image.open(color_frame_path)
        depth_frame = np.load(depth_frame_path)
        
        if color_frame is None or depth_frame is None:
            print("[Vision Router] Missing color or depth frame")
            raise HTTPException(status_code=500, detail="Missing color or depth frame.")

        # Query for all target classes
        query = f"Find {', '.join(target_classes)}"
        print(f"[Vision Router] Query: {query}")
        response = query_router(query, "camera", camera, color_frame, gemini, config)

        if not response.get("matches_found", False):
            print(f"[Vision Router] None of the target objects {target_classes} were found")
            raise HTTPException(status_code=404, detail=f"None of the target objects {target_classes} were found.")

        detected_objects = response.get("objects", [])
        results = []

        # Process each detected object
        for obj in detected_objects:
            centers = gemini.get_object_centers(color_frame, [obj], depth_frame, intrinsics)
            if centers and len(centers) > 0:
                print(f"[Vision Router] Found {len(centers)} centers for {obj}")
                for center in centers:
                    results.append(ObjectCenterResponse(
                        object_name=obj,
                        center_coordinates=center
                    ))
            else:
                print(f"[Vision Router] Object '{obj}' center not found")

        if not results:
            print("[Vision Router] No object centers were found")
            raise HTTPException(status_code=404, detail="No object centers were found.")

        print(f"[Vision Router] Returning {len(results)} object centers")
        return results

    except Exception as e:
        print(f"[Vision Router] Error in detect_object_centers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

