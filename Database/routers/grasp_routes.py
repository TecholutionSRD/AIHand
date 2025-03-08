from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config
from Database.functions.db_mannger import DatabaseManager

##-----------------------------------------------------#
grasp_router = APIRouter(prefix="/grasp_db", tags=["Grasp DB"])

def get_database():
    return DatabaseManager("db_config.yaml")

##-----------------------------------------------------#
class GraspRequest(BaseModel):
    object_name: str
    grasp_distance: float
    pickup_mode: str

@grasp_router.post("/create", response_model=Dict)
async def create_grasp(grasp: GraspRequest):
    """Create a grasp in the database."""
    db = get_database()
    response = await db.grasp_orchestrator("create", object_name=grasp.object_name, grasp_distance=grasp.grasp_distance, pickup_mode=grasp.pickup_mode)
    if response:
        return response
    raise HTTPException(status_code=500, detail="Failed to create grasp")

@grasp_router.get("/get")
async def get_specific_grasp(grasp_id: str):
    """Fetch a specific grasp by ID."""
    db = get_database()
    response = await db.grasp_orchestrator("read", grasp_id=grasp_id)
    if response and response["status"] == "success":
        return response
    raise HTTPException(status_code=404, detail="Grasp not found")

@grasp_router.put("/update", response_model=Dict)
async def update_grasp(grasp_id: str, grasp: GraspRequest):
    """Update a grasp by ID."""
    db = get_database()
    response = await db.grasp_orchestrator("update", grasp_id=grasp_id, object_name=grasp.object_name, grasp_distance=grasp.grasp_distance, pickup_mode=grasp.pickup_mode)
    if response and response["status"] == "success":
        return response
    raise HTTPException(status_code=404, detail="Grasp not found")

@grasp_router.delete("/delete", response_model=Dict)
async def delete_grasp(grasp_id: str):
    """Delete a grasp by ID."""
    db = get_database()
    response = await db.grasp_orchestrator("delete", grasp_id=grasp_id)
    if response and response["status"] == "success":
        return response
    raise HTTPException(status_code=404, detail="Grasp not found")
##-----------------------------------------------------#