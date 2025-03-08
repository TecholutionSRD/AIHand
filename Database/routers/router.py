"""
This file contains the routes for the database microservice.
"""
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

@db_router.get("/connect_db", response_model=Dict)
async def connect_db():
    """Connect to the database"""
    db = get_database()
    if db.client:
        return {"status": "success", "message": "Connected to database"}
    raise HTTPException(status_code=500, detail="Failed to connect to database")

@db_router.get("/close_db", response_model=Dict)
async def close_db():
    """Close the database connection"""
    db = get_database()
    db.close()
    return {"status": "success", "message": "Database connection closed"}

@db_router.get("/create_db", response_model=Dict)
async def create_db():
    """Create the database and collections"""
    db = get_database()
    if db.create_database() and db.create_collections():
        return {"status": "success", "message": "Database and collections created"}
    raise HTTPException(status_code=500, detail="Failed to create database or collections")
#-------------------------------------------------------------------------------------------------------#

