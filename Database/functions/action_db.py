"""
This file contains CRUD functions for the action_db database.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from bson import ObjectId
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config

class ActionDB:
    """
    This class provides CRUD operations for actions in a MongoDB database.
    Attributes:
        config (dict): Configuration settings for the database loaded from config file
        collection: MongoDB collection object for actions_db
    Methods:
        create_action(action_data: Dict) -> str:
            Creates a new action record in the database
        get_action(action_id: str) -> Optional[Dict]:
            Retrieves an action by its ID
        update_action(action_id: str, action_data: Dict) -> bool:
            Updates an existing action with new data
        delete_action(action_id: str) -> bool:
            Deletes an action from the database
        list_actions() -> List[Dict]:
            Retrieves all actions from the database
    """
    def __init__(self, config_path:Path, db):
        """
        Initializes the ActionDB object.
            Args:
                config_path (Path): The path to the configuration file.
                db: The database object.
        """
        config = load_config(config_path)
        self.config = config.get('Database',{})
        self.collection = db['actions_db']

    async def create_action(self, action_data: Dict) -> str:
        """
        Create a new action record
        Args:
            action_data (Dict) : A dictionary containing all the actions_db detailed request.
        """
        try:
            result = self.collection.insert_one(action_data)
            action_id = str(result.inserted_id)
            return action_id
        except Exception as e:
            raise Exception(f"Failed to create action: {str(e)}")

    async def get_action(self, action_id: str) -> Optional[Dict]:
        """
        Retrieve an action by ID
        """
        try:
            action = self.collection.find_one({"_id": ObjectId(action_id)})
            if action:
                # Convert ObjectId to string before returning
                action['_id'] = str(action['_id'])
                return action
            return None
        except Exception as e:
            raise Exception(f"Failed to retrieve action: {str(e)}")

    async def update_action(self, action_id: str, action_data: Dict) -> bool:
        """
        Update an existing action
        """
        try:
            result = self.collection.update_one(
                {"_id": ObjectId(action_id)},
                {"$set": action_data}
            )
            return result.modified_count > 0
        except Exception as e:
            raise Exception(f"Failed to update action: {str(e)}")

    async def delete_action(self, action_id: str) -> bool:
        """
        Delete an action by ID
        """
        try:
            result = self.collection.delete_one({"_id": ObjectId(action_id)})
            return result.deleted_count > 0
        except Exception as e:
            raise Exception(f"Failed to delete action: {str(e)}")

    async def list_actions(self) -> List[Dict]:
        """
        List all actions with serialized ObjectIds
        """
        try:
            cursor = self.collection.find()
            actions = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                actions.append(doc)
            return actions
        except Exception as e:
            raise Exception(f"Failed to list actions: {str(e)}")
