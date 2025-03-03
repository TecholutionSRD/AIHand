from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from Database.utils.db import *
from Config.config import load_config
#-------------------------------------------------------------------#
CONFIG_PATH = "db_config.yaml"

db_router = APIRouter(prefix="/database")

config = load_config(CONFIG_PATH)
db_config = config.get("DataBase", {})
#-------------------------------------------------------------------#
class GraspData(BaseModel):
    name: str
    grasp_distance: float
    pickup_mode: str

class ActionData(BaseModel):
    name: str
    samples: int
    model: Optional[bool] = True

class ActionObjectData(BaseModel):
    action_id: int
    object_name: str
#-------------------------------------------------------------------#
@db_router.get("/health")
async def health():
    """Check the health of the Database microservice."""
    print("[DB Router] Health check requested.")
    return {"status": "OK"}

#-------------------------------------------------------------------#
@db_router.get("/check_knowledgebase")
async def check_knowledgebase_route():
    """
    Endpoint to check the existence and validity of the knowledge base files.
    """
    try:
        print("[DB Router] Checking knowledge base.")
        grasp_names, action_names = check_knowledgebase(db_config)
        return {"grasp_names": grasp_names, "action_names": action_names}
    except Exception as e:
        print(f"[DB Router] Error checking knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-------------------------------------------------------------------#
@db_router.get("/check_knowledge/{name}")
async def specific_knowledge_check_route(name: str):
    """
    Endpoint to check the existence of specific object knowledge in the database.
    """
    try:
        print(f"[DB Router] Checking knowledge for object: {name}")
        result = specific_knowledge_check(db_config, name)
        return result
    except Exception as e:
        print(f"[DB Router] Error checking knowledge for '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-------------------------------------------------------------------#
# @db_router.get("/action_objects/{action_name}")
async def get_action_objects_route(action_name: str):
    """
    Endpoint to retrieve objects associated with a specific action.
    """
    try:
        print(f"[DB Router] Retrieving objects for action: {action_name}")
        objects = get_action_objects(db_config, action_name)
        return {"action_name": action_name, "objects": objects}
    except Exception as e:
        print(f"[DB Router] Error retrieving objects for action '{action_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
#-------------------------------------------------------------------#
# @db_router.post("/add_grasp")
async def add_grasp_route(grasp_data: GraspData):
    """
    Endpoint to add grasp data to the database.
    """
    try:
        print(f"[DB Router] Adding grasp entry: {grasp_data}")
        success = add_grasp_data(db_config, grasp_data.name, grasp_data.grasp_distance, grasp_data.pickup_mode)
        if not success:
            print(f"[DB Router] Failed to add grasp entry for '{grasp_data.name}'.")
            raise HTTPException(status_code=400, detail="Grasp entry already exists or invalid input.")
        print(f"[DB Router] Successfully added grasp entry for '{grasp_data.name}'.")
        return {"message": f"Grasp entry for '{grasp_data.name}' added successfully."}
    except Exception as e:
        print(f"[DB Router] Error adding grasp entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-------------------------------------------------------------------#
# @db_router.post("/add_action")
async def add_action_route(action_data: ActionData):
    """
    Endpoint to add action data to the database.
    """
    try:
        print(f"[DB Router] Adding action entry: {action_data}")
        action_id = add_action_data(db_config, action_data.name, action_data.samples, action_data.model)
        if action_id is None:
            print(f"[DB Router] Failed to add action entry for '{action_data.name}'.")
            raise HTTPException(status_code=400, detail="Action entry already exists or invalid input.")
        print(f"[DB Router] Successfully added action entry for '{action_data.name}' with ID {action_id}.")
        return {"message": f"Action entry for '{action_data.name}' added successfully.", "id": action_id}
    except Exception as e:
        print(f"[DB Router] Error adding action entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-------------------------------------------------------------------#
# @db_router.post("/add_action_object")
async def add_action_object_route(action_object_data: ActionObjectData):
    """
    Endpoint to associate an object with an action in the database.
    """
    try:
        print(f"[DB Router] Adding association: Action ID {action_object_data.action_id} -> Object '{action_object_data.object_name}'")
        success = add_action_object(db_config, action_object_data.action_id, action_object_data.object_name)
        if not success:
            print(f"[DB Router] Failed to add association for action ID {action_object_data.action_id}.")
            raise HTTPException(status_code=400, detail="Association already exists or invalid input.")
        print(f"[DB Router] Successfully added association for action ID {action_object_data.action_id} and object '{action_object_data.object_name}'.")
        return {"message": f"Association between action ID {action_object_data.action_id} and object '{action_object_data.object_name}' added successfully."}
    except Exception as e:
        print(f"[DB Router] Error adding action-object association: {e}")
        raise HTTPException(status_code=500, detail=str(e))
#-------------------------------------------------------------------#

