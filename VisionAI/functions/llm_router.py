import json
import os
import sys
import json_repair
from typing import List, Dict, Optional
import re
from difflib import get_close_matches
from dotenv import load_dotenv
import google.generativeai as genai

def get_action_objects(db_config: dict, action_name: str) -> List[str]:
    """
    Get valid objects associated with a specific action from the database config.
    
    Args:
        db_config: Database configuration dictionary
        action_name: Name of the action to look up
    
    Returns:
        List of object names valid for the action
    """
    try:
        # Assuming db_config has a structure with actions and their valid objects
        actions_config = db_config.get('actions', {})
        action_data = actions_config.get(action_name, {})
        return action_data.get('valid_objects', [])
    except Exception as e:
        print(f"Error getting action objects: {str(e)}")
        return []

from Database.functions.db_mannger import DatabaseManager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
# from Database_old.functions.db import check_knowledgebase, get_action_objects
from Config.config import load_config

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# --- IMPROVED FUNCTION TO CLEAN OBJECT NAMES ---
def normalize_object_name(name: str) -> str:
    """
    Normalize object names by extracting key components and removing variations.
    
    Examples:
    - "a small red ketchup bottle" -> "ketchup bottle"
    - "bottle of ketchup" -> "ketchup bottle"
    - "a piece of toast on a white plate" -> "toast"
    - "large silver spoon" -> "spoon"
    """
    name = name.lower()
    
    pattern = re.search(r"(\w+)\s+of\s+(\w+)", name)
    if pattern:
        container, content = pattern.groups()
        name = f"{content} {container}"
    
    cleaned_name = re.sub(r"\b(small|large|piece|of|on|a|the|an|white|blue|red|black|green|yellow)\b", "", name, flags=re.IGNORECASE).strip()
    
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    
    cleaned_name = re.sub(r"pair of (\w+)", r"\1", cleaned_name)
    
    return cleaned_name

# --- UPDATED OBJECT MATCHING FUNCTION WITH FUZZY MATCHING ---
def find_closest_matches(query_items: List[str], detected_items: List[str]) -> List[str]:
    """
    Matches user query objects to detected objects using fuzzy matching and normalization.
    
    Args:
        query_items: List of objects requested by the user
        detected_items: List of objects detected in the scene
    
    Returns:
        List of actual detected objects that match the query
    """
    normalized_detected = {obj: normalize_object_name(obj) for obj in detected_items}
    normalized_query = [normalize_object_name(q) for q in query_items]
    
    matches = []
    # Check for each query item
    for query_norm in normalized_query:
        current_matches = []
        for detected_obj, detected_norm in normalized_detected.items():
            # Direct match
            if query_norm == detected_norm:
                current_matches.append(detected_obj)
            # Substring match
            elif query_norm in detected_norm or detected_norm in query_norm:
                current_matches.append(detected_obj)
            else:
                # Keyword matching
                query_keywords = set(query_norm.split())
                detected_keywords = set(detected_norm.split())
                common_words = query_keywords.intersection(detected_keywords)
                if common_words and not common_words.issubset({"a", "the", "of", "on", "in"}):
                    current_matches.append(detected_obj)
        
        # Add all matches found for this query item
        matches.extend(current_matches)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(matches))

# --- UPDATED QUERY ROUTER ---
async def query_router(user_query: str, mode: str, camera, color_frame=None, gemini=None, config=None) -> Dict:
    """
    Maps user queries to available actions and objects using Gemini LLM.
    
    Args:
        user_query (str): Natural language query from user.
        mode (str): Either 'database' or 'camera' to determine matching source.
        camera: Optional camera instance for live detection.
        gemini: Optional Gemini instance for live detection.
        config: Configuration dictionary.
    """
    config = load_config("vision_ai_config.yaml")['Gemini']
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(config.get("model_name", 'gemini-1.5-flash-002'))

    if mode == "database":
        print(f"[LLM Router] Database mode")
        system_prompt = """Extract the action and objects from the user query.
        Format response as JSON:
        {
            "action": "<action_verb>",
            "objects": ["<object1>", "<object2>", ...]
        }
        Extract the user input **exactly** as given. Do not change the wording.
        """
    elif mode == "camera":
        system_prompt = """Extract only object names from the user query that could be visible in the camera.
            Format response as JSON:
            {
                "objects": ["<object1>", "<object2>", ...]
            }
            Extract the user input **exactly** as given. Do not change the wording.
            """
    else:
        return {"error": "Invalid mode. Use 'database' or 'camera'"}

    try:
        inputs = f"Prompt : {system_prompt}. \n User Input : {user_query}. \n Your Answer :"
        response = model.generate_content(inputs)
        result = json.loads(json_repair.repair_json(response.text))
        print(f"[LLM Router] User Output: {result}")

        if mode == "database":
            db_configs = load_config('db_config.yaml')
            db_config = db_config.get("DataBase", {})
            db_manager = DatabaseManager('db_config.yaml')
            
            # Get all actions and grasps
            actions_response = await db_manager.action_orchestrator("list")
            grasps_response = await db_manager.grasp_orchestrator("list")
            
            # Extract names from the response data
            action_names = [action["action_name"] for action in actions_response.get("data", [])]
            grasp_names = [grasp["object_name"] for grasp in grasps_response.get("data", [])]
            
            # Close database connection
            db_manager.close()
            print(f"[LLM Router] Grasp Names: {grasp_names}")
            print(f"[LLM Router] Action Names: {action_names}")

            action = result.get("action", "")
            matched_actions = find_closest_matches([action], action_names)
            print(f"[LLM Router] Matched Actions: {matched_actions}")

            if matched_actions:
                matched_action = matched_actions[0]
                valid_objects = get_action_objects(db_config, matched_action)
                print(f"[LLM Router] Valid Objects for Action '{matched_action}': {valid_objects}")

                query_objects = result.get("objects", [])
                matched_objects = find_closest_matches(query_objects, valid_objects)
                print(f"[LLM Router] Matched Objects: {matched_objects}")

                return {
                    "action": matched_action,
                    "objects": matched_objects,
                    "matches_found": bool(matched_objects)
                }
            return {"error": "No matching action found in database"}

        elif mode == "camera":
            if not camera and not color_frame:
                return {"error": "Camera instance or color frame required for camera mode"}

            query_objects = result.get("objects", [])

            detected_objects = gemini.detect(color_frame)

            matched_objects = find_closest_matches(query_objects, detected_objects)
            print(f"[LLM Router] Matched Objects: {matched_objects}")

            return {
                "objects": matched_objects,
                "matches_found": bool(matched_objects)
            }

        else:
            return {"error": "Invalid mode. Use 'database' or 'camera'"}

    except Exception as e:
        print(f"[LLM Router] Error processing query: {str(e)}")
        return {"error": f"Error processing query: {str(e)}"}
    
async def validate_task(action_name: str, object_names: List[str]) -> Dict:
    """
    Check action and grasp details from database for given action and objects.
    
    Args:
        action_name: Name of the action to look up
        object_names: List of object names to check
    
    Returns:
        Dictionary containing action details, grasp details and status message
    """
    try:
        db_manager = DatabaseManager('db_config.yaml')
        
        # Check action in database
        action_response = await db_manager.action_orchestrator("get", {"action_name": action_name})
        
        if not action_response.get("data"):
            db_manager.close()
            return {
                "status": "error",
                "message": f"Action '{action_name}' not found in database"
            }
            
        action_details = action_response["data"]
        grasp_details = []
        
        # Check grasp points for each object
        for obj_name in object_names:
            grasp_response = await db_manager.grasp_orchestrator("get", {"object_name": obj_name})
            if grasp_response.get("data"):
                grasp_details.append(grasp_response["data"])
                
        db_manager.close()
        
        return {
            "status": "success",
            "action_details": action_details,
            "grasp_details": grasp_details,
            "message": "Successfully retrieved action and grasp details"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking action and grasp details: {str(e)}"
        }