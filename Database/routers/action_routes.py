from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config
from Database.functions.db_mannger import DatabaseManager

##-----------------------------------------------------#
action_router = APIRouter(prefix="/action_db", tags=["Action DB"])

def get_database():
    return DatabaseManager("db_config.yaml")
##-----------------------------------------------------#

# Action Request Model
class ActionRequest(BaseModel):
    action_name: str
    tool: str
    object: str

@action_router.post("/create", response_model=Dict)
async def create_action(action: ActionRequest):
    """Create an action in the database."""
    db = get_database()
    response = await db.action_orchestrator("create", action_name=action.action_name, model_available=False, tool=action.tool, object=action.object)
    if response:
        return response
    raise HTTPException(status_code=500, detail="Failed to create action")

@action_router.get("/get")
async def get_specific_action(action_id: str):
    """Fetch a specific action by ID."""
    db = get_database()
    response = await db.action_orchestrator("read", action_id=action_id)
    if response and response["status"] == "success":
        return response
    raise HTTPException(status_code=404, detail="Action not found")

@action_router.put("/update", response_model=Dict)
async def update_action(action_id: str, action: ActionRequest):
    """Update an action by ID."""
    db = get_database()
    response = await db.action_orchestrator("update", action_id=action_id, action_name=action.action_name, tool=action.tool, object=action.object)
    if response and response["status"] == "success":
        return response
    raise HTTPException(status_code=404, detail="Action not found")

@action_router.delete("/delete", response_model=Dict)
async def delete_action(action_id: str):
    """Delete an action by ID."""
    db = get_database()
    response = await db.action_orchestrator("delete", action_id=action_id)
    if response and response["status"] == "success":
        return response
    raise HTTPException(status_code=404, detail="Action not found")
##-----------------------------------------------------#