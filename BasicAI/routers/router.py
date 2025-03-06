"""
This file contains the routes for BasicAI microservice.
"""
import os
import asyncio
from fastapi import APIRouter, HTTPException
from functools import lru_cache
from pydantic import BaseModel
from BasicAI.functions.Inference.inference import realtime_inference
from BasicAI.functions.dataset.preprocessor import PreProcessor
from VisionAI.functions.vision_detection import GeminiInference
from Camera.functions.camera_reciver import CameraReceiver
from Config.config import load_config
#-------------------------------------------------------------------#
CONFIG_PATH = "basic_ai_config.yaml"
CAMERA_CONFIG = "camera_config.yaml"
VISION_CONFIG = "vision_ai_config.yaml"
ai_router = APIRouter(prefix="/AI")
config = load_config(CONFIG_PATH)
vision_config = load_config(VISION_CONFIG)
#-------------------------------------------------------------------#
@lru_cache()
def get_processor():
    """Cache Video Processor Innstance to avoid repeated instantiation"""
    return PreProcessor(CONFIG_PATH)

@lru_cache()
def get_gemini():
    """Cache Gemini Model instance to avoid repeated loading."""
    return GeminiInference(vision_config)

@lru_cache()
def get_camera_receiver():
    """Cache CameraReceiver instance to avoid repeated instantiation."""
    return CameraReceiver(CAMERA_CONFIG)
#-------------------------------------------------------------------#
@ai_router.get("/health")
async def health_check():
    return {"status": "OK"}

#--------------------------------------------------------------------#
@ai_router.post("/Hamer")
async def process_hamer(action_name: str, objects: list[str]):
    """
    Process Hamer with given objects and action name
    Args:
        action_name (str): Name of the action
        objects (list[str]): List of objects to process 
    Returns:
        dict: Processing result
    """
    try:
        processor = get_processor()
        result = processor.process_hamer(action_name, objects)
        return result
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"[AI Router] Hamer processing failed: {str(e)}")
#-----------------------------------------------------------------#
class ResponseProcessor(BaseModel):
    """
    This class represents the response of the endpoint preprocessor.
    """
    message: str
    status: bool
    action_name: str

@ai_router.post("/preprocess")
async def preprocess_action(action_name: str):
    """
    Endpoint to preprocess action data by filtering, renaming, and merging CSV files. 
    Args:
        action_name (str): The name of the action to preprocess  
    Returns:
        ResponseProcessor: Contains processing status and details 
    Raises:
        HTTPException: If preprocessing fails
    """
    try:
        preprocessor = get_processor()
        result = preprocessor.process(action_name)
        if result:
            return ResponseProcessor(message="[AI Router] Data preprocessing completed successfully",status=result,action_name=action_name)
        else:
            return ResponseProcessor(message="[AI Router] Action not found !",status=result,action_name=action_name)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[AI Router] Preprocessing failed for action {action_name}: {str(e)}")
#-----------------------------------------------------------------#
@ai_router.post("/inference")
async def run_inference(target_classes: list[str]):
    """
    Endpoint to run the inference pipeline with object detection and trajectory prediction.

    Args:
        target_classes (list[str]): List of target classes to detect

    Returns:
        dict: Path to the output trajectory file or error message
    """
    try:
        camera = get_camera_receiver()
        gemini = get_gemini()
        print("[AI Router] Initialized Models")

        target_classes = [cls.lower().strip() for cls in target_classes] 

        output_path = await realtime_inference(camera, gemini, target_classes)

        if output_path:
            return {"status": "success", "output_path": output_path}
        else:
            raise HTTPException(status_code=500, detail="[AI Router] Trajectory inference failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[AI Router] Inference pipeline failed: {str(e)}")