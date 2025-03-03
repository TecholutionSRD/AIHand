"""
This file contains the code to map input comment from user to the available objects or actions we have infront of us.
This allows us to be safe and not send errors of object,action name not matching.
"""

import json
import os
import threading
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
import sys

import json_repair

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from Database.functions.db import check_knowledgebase, get_action_objects
from Config.config import load_config

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

def find_closest_matches(query_items: List[str], available_items: List[str], model: genai.GenerativeModel) -> List[str]:
    """
    Find the closest matching items from a list of available items based on a list of query items using a generative model.
    Args:
        query_items (List[str]): A list of items to find matches for.
        available_items (List[str]): A list of items to match against.
        model (genai.GenerativeModel): A generative model used to find semantic matches.
    Returns:
        List[str]: A list of closest matching items from the available items. If no matches are found, an empty list is returned.
    
    The function follows these rules:
    1. Only return items from the available list.
    2. Return exact matches if they exist.
    3. Return the closest semantic match if no exact match exists.
    4. If no reasonable match exists, return an empty list.
    Raises:
        Exception: If there is an error during the matching process, it prints the error and returns an empty list.
    
    if not query_items or not available_items:
        return []
    """   
    matching_prompt = f"""
    From the available items: {available_items}
    Find the closest matching items for: {query_items}
    
    Rules:
    1. Only return items from the available list
    2. Return exact matches if they exist
    3. Return the closest semantic match if no exact match exists
    4. If no reasonable match exists, return empty
    
    Format response as JSON:
    {{"matches": ["matched_item1", "matched_item2", ...]}}
    """
    
    try:
        response = model.generate_content(matching_prompt)
        result = json.loads(json_repair.repair_json(response.text))
        return result.get("matches", [])
    except Exception as e:
        print(f"Error in matching: {e}")
        return []

async def query_router(user_query: str, mode: str, camera=None, gemini=None, config=None) -> Dict:
    """
    Maps user queries to available actions and objects using Gemini LLM.
    
    Args:
        user_query (str): Natural language query from user
        mode (str): Either 'database' or 'camera' to determine matching source
        camera: Optional camera instance for live detection
        gemini: Optional gemini instance for live detection
        config: Configuration dictionary
    """
    # Configure Gemini
    config = load_config("vision_ai_config.yaml")['Gemini']
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(config.get("model_name", 'gemini-1.5-flash-002'))
    print(f"[LLM Router] Model Loaded")
    
    # Prepare system prompt based on mode
    if mode == "database":
        print(f"[LLM Router] Database mode")
        system_prompt = """Extract the action and objects from the user query.
        Format response as JSON:
        {
            "action": "<action_verb>",
            "objects": ["<object1>", "<object2>", ...]
        }
        Extract the user input very accuratly do not change them.
        """
    elif mode == "camera":
        print(f"[LLM Router] Camera Mode")
        system_prompt = """Extract only object names from the user query that could be visible in camera.
            Format response as JSON:
            {
                "objects": ["<object1>", "<object2>", ...]
            }
            Extract the user input very accuratly do not change them.
            """
    else:
        return {"error": "Invalid mode. Use 'database' or 'camera'"}

    try:
        inputs = f"Prompt : {system_prompt}. \n User Input : {user_query}. \n Your Answer :"
        response = model.generate_content(inputs)
        result = json.loads(json_repair.repair_json(response.text))
        print(f"[LLM Router] User Output: {result}")
        
        if mode == "database":
            db_config = config.get("DataBase", {})
            grasp_names, action_names = check_knowledgebase(db_config)
            print(f"[LLM Router] Grasp Names: {grasp_names}")
            print(f"[LLM Router] Action Names: {action_names}")
            
            # Find closest action match using Gemini
            action = result.get("action", "")
            matched_actions = find_closest_matches([action], action_names, model)
            print(f"[LLM Router] Matched Actions: {matched_actions}")
            
            if matched_actions:
                matched_action = matched_actions[0]
                valid_objects = get_action_objects(db_config, matched_action)
                print(f"[LLM Router] Valid Objects for Action '{matched_action}': {valid_objects}")
                
                # Find closest object matches using Gemini
                query_objects = result.get("objects", [])
                matched_objects = find_closest_matches(query_objects, valid_objects, model)
                print(f"[LLM Router] Matched Objects: {matched_objects}")
                
                return {
                    "action": matched_action,
                    "objects": matched_objects,
                    "matches_found": True
                }
            return {"error": "No matching action found in database"}
            
        elif mode == "camera":
            if not camera:
                return {"error": "Camera instance required for camera mode"}
                
            query_objects = result.get("objects", [])
            print(f"[LLM Router] Query Objects: {query_objects}")
            detected_objects = await gemini.detect_all(camera)
            print(f"[LLM Router] Detected Objects: {detected_objects}")
            
            # Find closest matches using Gemini
            matched_objects = find_closest_matches(query_objects, detected_objects, model)
            print(f"[LLM Router] Matched Objects: {matched_objects}")
            
            return {
                "objects": matched_objects,
                "matches_found": bool(matched_objects)
            }
            
        else:
            return {"error": "Invalid mode. Use 'database' or 'camera'"}
            
    except Exception as e:
        return {"error": f"Error processing query: {str(e)}"}
