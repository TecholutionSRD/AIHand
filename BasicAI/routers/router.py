"""
This file contains the routes for BasicAI microservice.
"""
import os
import asyncio
from fastapi import APIRouter, HTTPException
from functools import lru_cache
from pydantic import BaseModel
from BasicAI.functions.dataset.preprocessor import PreProcessor
from Config.config import load_config
#-------------------------------------------------------------------#
CONFIG_PATH = "basic_ai_config.yaml"
ai_router = APIRouter(prefix="/AI")
config = load_config(CONFIG_PATH)
#-------------------------------------------------------------------#
@lru_cache()
def get_processor():
    """Cache Video Processor Innstance to avoid repeated instantiation"""
    return PreProcessor(CONFIG_PATH)
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
