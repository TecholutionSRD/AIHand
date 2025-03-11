"""
This file contains the routes for BasicAI microservice.
"""
import os
import asyncio
from fastapi import APIRouter, HTTPException
from functools import lru_cache
from pydantic import BaseModel
from BasicAI.functions.Inference.inference import inference, realtime_inference
from BasicAI.functions.dataset.preprocessor import PreProcessor
from VisionAI.functions.vision_detection import GeminiInference
from Camera.functions.camera_receiver import CameraReceiver
from BasicAI.functions.dataset.data_generator import VideoProcessor4D
from BasicAI.functions.trainer.trainer import train
from Config.config import load_config
import pandas as pd

#-------------------------------------------------------------------#
CONFIG_PATH = "basic_ai_config.yaml"
CAMERA_CONFIG = "camera_config.yaml"
VISION_CONFIG = "vision_ai_config.yaml"
COBOT_CONFIG
ai_router = APIRouter(prefix="/AI")
config = load_config(CONFIG_PATH)

#-------------------------------------------------------------------#
@lru_cache()
def get_preprocessor():
    """Cache PreProcessor instance to avoid repeated instantiation"""
    return PreProcessor(CONFIG_PATH)

@lru_cache()
def get_gemini_model():
    """Cache Gemini Model instance to avoid repeated loading."""
    return GeminiInference(VISION_CONFIG)

@lru_cache()
def get_camera_receiver():
    """Cache CameraReceiver instance to avoid repeated instantiation."""
    return CameraReceiver(CAMERA_CONFIG)

@lru_cache()
def get_cobot_client():
    return CobotClient(COBOT_CONFIG)
#-------------------------------------------------------------------#
@ai_router.get("/health")
async def health():
    """
    Health check endpoint to verify if all required components are loaded properly.
    
    Returns:
        dict: Status of the health check
    """
    try:
        _ = get_preprocessor()
        _ = get_gemini_model()
        _ = get_camera_receiver()
        return {"status": "OK"}
    except Exception as e:
        return {"status": "ERROR", "detail": str(e)}

#--------------------------------------------------------------------#
@ai_router.post("/process_action_data")
async def process_action_data(action_name: str, objects: list[str] = None, hamer_mode: bool = False):
    """
    Process action data based on the specified mode.
    
    Args:
        action_name (str): Name of the action to process.
        objects (list[str], optional): List of objects required for Hamer mode.
        hamer_mode (bool, optional): If True, processes action in Hamer mode (recording & augmentation). Defaults to False.

    Returns:
        dict: Processing status and result.
    """
    try:
        preprocessor = get_preprocessor()
        result = await preprocessor.process_action_data(action_name, objects, hamer=hamer_mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[AI Router] Action processing failed: {str(e)}")

#-----------------------------------------------------------------#
class PreprocessingResponse(BaseModel):
    """
    Response model for preprocessing endpoint.
    """
    message: str
    status: bool
    action_name: str

@ai_router.post("/preprocess_existing_data")
async def preprocess_existing_data(action_name: str):
    """
    Preprocess existing action data by filtering, renaming, and merging CSV files.
    
    Args:
        action_name (str): The name of the action to preprocess.

    Returns:
        PreprocessingResponse: Processing status and details.

    Raises:
        HTTPException: If preprocessing fails.
    """
    try:
        preprocessor = get_preprocessor()
        result = await preprocessor.process_action_data(action_name, hamer=False)

        return PreprocessingResponse(
            message="[AI Router] Data preprocessing completed successfully" if result["status"] == "success" else "[AI Router] Action not found!",
            status=result["status"] == "success",
            action_name=action_name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[AI Router] Preprocessing failed for action {action_name}: {str(e)}")

#-----------------------------------------------------------------#
# @ai_router.post("/run_inference")
# async def run_inference(target_classes: list[str]):
#     """
#     Run inference pipeline with object detection and trajectory prediction.
    
#     Args:
#         target_classes (list[str]): List of target classes to detect.

#     Returns:
#         dict: Path to the output trajectory file or error message.
#     """
#     try:
#         camera = get_camera_receiver()
#         gemini = get_gemini_model()
#         print("[AI Router] Initialized Models")

#         target_classes = [cls.lower().strip() for cls in target_classes] 

#         output_path = await realtime_inference(camera, gemini, target_classes)

#         if output_path:
#             return {"status": "success", "output_path": output_path}
#         else:
#             raise HTTPException(status_code=500, detail="[AI Router] Trajectory inference failed")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"[AI Router] Inference pipeline failed: {str(e)}")

#-----------------------------------------------------------------#
@ai_router.post("/process_all_videos")
async def process_all_videos(action_name: str):
    """
    Process all videos for a given action.
    
    Args:
        action_name (str): Name of the action to process.

    Returns:
        dict: Processing status.
    """
    try:
        video_processor = VideoProcessor4D(CONFIG_PATH)
        video_processor.process_all_videos(action_name)
        
        return {"message": f"[AI Router] Successfully processed videos for action {action_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[AI Router] Video processing failed for action {action_name}: {str(e)}")

#-------------------------------------------------------------------#
@ai_router.post("/train_model")
def train_model(action_name: str):
    """
    Train the model for a given action.

    Args:
        action_name (str): Name of the action to train the model on.

    Returns:
        dict: Status message of the training process.

    Raises:
        HTTPException: If training fails.
    """
    dataset_path = f"data/recordings/{action_name}/dataset.csv"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"[AI Router] Dataset not found for action {action_name}")

    try:
        dataset = pd.read_csv(dataset_path)
        input_dim = 6
        output_dim = dataset.shape[1] - input_dim
        train(CONFIG_PATH, input_dim, output_dim, dataset_path)

        return {"message": f"[AI Router] Model training completed successfully for action {action_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[AI Router] Model training failed for action {action_name}: {str(e)}")
#-------------------------------------------------------------------#
@ai_router.post("/run_inference")
async def run_inference(action_name:str, target_classes: list[str]):
    """
    Run inference pipeline with object detection and trajectory prediction.
    
    Args:
        target_classes (list[str]): List of target classes to detect.

    Returns:
        dict: Path to the output trajectory file or error message.
    """
    try:
        camera = get_camera_receiver()
        gemini = get_gemini_model()
        print("[AI Router] Initialized Models")

        target_classes = [cls.lower().strip() for cls in target_classes] 

        response = await inference(camera, gemini, action_name, target_classes)

        if response:
            payload = response.get('payload',{})
            client = get_cobot_client()
            client.send_trajectory_data(payload)
            return {"status": "success", "message": "Inference completed and trajectory sent", "payload": payload}
        else:
            raise HTTPException(status_code=500, detail="[AI Router] Inference failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[AI Router] Inference pipeline failed: {str(e)}")
        