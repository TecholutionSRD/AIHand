"""
This file contains CRUD functions for the grasp_db database.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from bson import ObjectId
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config

class GraspDB:
    """
    This class provides CRUD operations for grasps in a MongoDB database.
    Attributes:
        config (dict): Configuration settings for the database loaded from config file
        collection: MongoDB collection object for grasp_db
    Methods:
        create_grasp(grasp_data: Dict) -> str:
            Creates a new grasp record in the database
        get_grasp(grasp_id: str) -> Optional[Dict]:
            Retrieves a grasp by its ID
        update_grasp(grasp_id: str, grasp_data: Dict) -> bool:
            Updates an existing grasp with new data
        delete_grasp(grasp_id: str) -> bool:
            Deletes a grasp from the database
        list_grasps() -> List[Dict]:
            Retrieves all grasps from the database
    """
    def __init__(self, config_path:Path, db):
        """
        Initializes the GraspDB object.
            Args:
                config_path (Path): The path to the configuration file.
                db: The database object.
        """
        config = load_config(config_path)
        self.config = config.get('Database',{})
        self.collection = db['grasp_db']

    async def create_grasp(self, grasp_data: Dict) -> str:
        """
        Create a new grasp record
        Args:
            grasp_data (Dict) : A dictionary containing all the grasp_db detailed request.
        """
        try:
            result = self.collection.insert_one(grasp_data)
            grasp_id = str(result.inserted_id)
            return grasp_id
        except Exception as e:
            raise Exception(f"Failed to create grasp: {str(e)}")

    async def get_grasp(self, grasp_id: str) -> Optional[Dict]:
        """
        Retrieve a grasp by ID
        """
        try:
            grasp = self.collection.find_one({"_id": ObjectId(grasp_id)})
            if grasp:
                # Convert ObjectId to string before returning
                grasp['_id'] = str(grasp['_id'])
                return grasp
            return None
        except Exception as e:
            raise Exception(f"Failed to retrieve grasp: {str(e)}")

    async def update_grasp(self, grasp_id: str, grasp_data: Dict) -> bool:
        """
        Update an existing grasp
        """
        try:
            result = self.collection.update_one(
                {"_id": ObjectId(grasp_id)},
                {"$set": grasp_data}
            )
            return result.modified_count > 0
        except Exception as e:
            raise Exception(f"Failed to update grasp: {str(e)}")

    async def delete_grasp(self, grasp_id: str) -> bool:
        """
        Delete a grasp by ID
        """
        try:
            result = self.collection.delete_one({"_id": ObjectId(grasp_id)})
            return result.deleted_count > 0
        except Exception as e:
            raise Exception(f"Failed to delete grasp: {str(e)}")

    async def list_grasps(self) -> List[Dict]:
        """
        List all grasps with serialized ObjectIds
        """
        try:
            cursor = self.collection.find()
            grasps = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                grasps.append(doc)
            return grasps
        except Exception as e:
            raise Exception(f"Failed to list grasps: {str(e)}")