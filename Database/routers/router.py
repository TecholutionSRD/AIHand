"""
This file contains the routes for the database microservice.
"""
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config
from Database.functions.db_mannger import DatabaseManager
#-------------------------------------------------------------------------------------------------------#
CONFIG_PATH = "db_config.yaml"

db_router = APIRouter(prefix="/database")

config = load_config(CONFIG_PATH)
db_config = config.get("DataBase", {})
#-------------------------------------------------------------------------------------------------------#
def get_database():
    return DatabaseManager(CONFIG_PATH)

#-------------------------------------------------------------------------------------------------------#
# These should be called internally

# @db_router.get("/connect_db", response_model=Dict)
async def connect_db():
    """Connect to the database"""
    db = get_database()
    if db.client:
        return {"status": "success", "message": "Connected to database"}
    raise HTTPException(status_code=500, detail="Failed to connect to database")

# @db_router.get("/close_db", response_model=Dict)
async def close_db():
    """Close the database connection"""
    db = get_database()
    db.close()
    return {"status": "success", "message": "Database connection closed"}

# @db_router.get("/create_db", response_model=Dict)
async def create_db():
    """Create the database and collections"""
    db = get_database()
    if db.create_database() and db.create_collections():
        return {"status": "success", "message": "Database and collections created"}
    raise HTTPException(status_code=500, detail="Failed to create database or collections")
#-------------------------------------------------------------------------------------------------------#
class VideoGalleryResponse(BaseModel):
    """
    Response model for the video gallery simulation API.

    This model structures the response containing lists of paths for sample directories,
    object files, CSV files, and video files.

    Attributes:
        action_name (str): Name of the action being queried
        sample_dirs (List[str]): List of sample directory paths
        object_files (List[str]): List of object file paths
        csv_files (List[str]): List of prediction CSV file paths
        video_files (List[str]): List of video file paths
    """
    action_name: str
    sample_dirs: List[str]
    object_files: List[str] 
    csv_files: List[str]
    video_files: List[str]

@db_router.get("/video_gallery", response_model=VideoGalleryResponse)
async def video_gallery_display(action_name: str):
    """
    Returns separate lists of paths for samples, objects, CSVs and videos.

    Args:
        action_name (str): Name of the action to query

    Returns:
        VideoGalleryResponse: Contains action name and separate lists for each file type

    Raises:
        HTTPException: If action directory doesn't exist or other errors occur
    """
    try:
        print("[DB Router] Hitting Video Gallery")
        action_name = str(action_name).lower()
        # Construct base path from action name 
        base_path = f"data/recordings/{action_name}"

        if not os.path.exists(base_path):
            raise HTTPException(status_code=404, detail=f"No recordings found for action: {action_name}")

        sample_dirs = []
        object_files = []
        csv_files = []
        video_files = []

        for d in os.listdir(base_path):
            sample_dir = os.path.join(base_path, d)

            if os.path.isdir(sample_dir):
                sample_dirs.append(sample_dir)
                object_file_path = f"{sample_dir}/{action_name}/objects.txt"
                csv_file_path = f"{sample_dir}/predictions_hamer.csv"
                json_file_path = f"{sample_dir}/gcp_url.json"
                if os.path.exists(json_file_path):
                    try:
                        with open(json_file_path, 'r') as f:
                            data = json.load(f)
                            video_file_path = data.get('gcp_url')
                    except Exception as e:
                        print(f"Error reading GCP URL from {json_file_path}: {e}")
                        video_file_path = f"{sample_dir}/{action_name}_video.mp4"
                else:
                    video_file_path = f"{sample_dir}/{action_name}_video.mp4" 
                
                object_files.append(object_file_path)
                csv_files.append(csv_file_path)
                video_files.append(video_file_path)

        return VideoGalleryResponse(
            action_name=action_name,
            sample_dirs=sample_dirs,
            object_files=object_files,
            csv_files=csv_files,
            video_files=video_files
        )

    except Exception as e:
        print(f"[Error] : An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))