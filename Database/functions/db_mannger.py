from typing import Any, Dict, Optional
import pymongo
from pathlib import Path
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config
from Database.functions.action_db import ActionDB
from Database.functions.grasp_db import GraspDB

class DatabaseManager:
    """DatabaseManager is a class to manage MongoDB database connections and operations.
    
    Attributes:
        config (dict): Configuration dictionary for the database.
        connection_string (str): MongoDB connection string.
        database_name (str): Name of the database to connect to.
        client (pymongo.MongoClient): MongoDB client instance.
        db (pymongo.database.Database): MongoDB database instance.
    
    Methods:
        __init__(config_path=None, config_dict=None):
        _load_config(config_path):
            Load configuration from a YAML file.
        connect():
        close():
        create_database():
        create_collections():
    """
    def __init__(self, config_path:Path):
        """
        Initialize the database connection using the configuration file or dict.
        Args:
            config_path (Path, optional): Path to the YAML config file
            config_dict (dict, optional): Configuration dictionary
        """
        config = load_config(config_path)
        self.config = config.get('DataBase', {})
        self.connection_string = self.config.get('URL', 'mongodb://localhost:27017/')
        self.database_name = self.config.get('database_name', 'AIHand')
        self.client = None
        self.db = None
        self.action_db = None
        
        self.connect()
        if self.client:
            self.create_database()
            self.create_collections()
            self.action_db = ActionDB(config_path, self.db)
            self.grasp_db = GraspDB(config_path, self.db)

    def connect(self):
        """
        Establish connection to MongoDB.
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = pymongo.MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            print(f"[DB] Connected to MongoDB successfully at {self.connection_string}")
            return True
        except Exception as e:
            print(f"[DB] Connection Error: {e}")
            self.client = None
            return False

    def close(self):
        """
        Close the MongoDB connection.
        """
        if self.client:
            try:
                self.client.close()
                print("[DB] MongoDB connection closed")
            except Exception as e:
                print(f"[DB] Error closing connection: {e}")
            finally:
                self.client = None
                self.db = None

    def create_database(self):
        """
        Create or connect to the specified database.
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.db = self.client[self.database_name]
            print(f"[DB] Using database '{self.database_name}'")
            return True
        except Exception as e:
            print(f"[DB] Database Error: {e}")
            return False

    def create_collections(self):
        """
        Create collections with specified columns defined in the configuration file.
        Returns:
            bool: True if all collections created successfully, False otherwise
        """
        try:
            collections = self.config.get('collections', {})

            if not collections:
                print("[DB] No collections specified in the config")
                return False

            for collection_name, columns in collections.items():
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
                    print(f"[DB] Collection '{collection_name}' created")
                    
                    validator = {
                        '$jsonSchema': {
                            'bsonType': 'object',
                            'required': list(columns.keys()),
                            'properties': {
                                field: {'type': dtype} for field, dtype in columns.items()
                            }
                        }
                    }
                    self.db.command('collMod', collection_name, validator=validator)
                    print(f"[DB] Schema validated for collection '{collection_name}'")
                else:
                    print(f"[DB] Collection '{collection_name}' already exists")
            return True
        except Exception as e:
            print(f"[DB] Collection Error: {e}")
            return False

    async def action_orchestrator(self, operation: str, action_id: Optional[str] = None, action_name: Optional[str] = None, 
                        tool: Optional[str] = None, model_available: Optional[bool] = False, 
                        object: Optional[str] = None) -> Dict[str, Any]:
        """
        This function acts as an orchestrator for the action database functions.
        
        Args:
            operation (str): A variable to define which CRUD operation to perform in the database.
        
        Returns:
            dict: A standardized response JSON.
        """
        if not self.action_db:
            raise Exception("ActionDB is not initialized")
        
        response = {"operation": operation,"status": "failed","message": "","action_id": action_id,"action_name": action_name,"model_available": model_available,"tool": tool,"object": object,"data": None}
        
        try:
            if operation == "create":
                action_data = {"action_name": action_name, "model_available": model_available, "tool": tool, "object": object}
                action_id = await self.action_db.create_action(action_data)
                response.update({"status": "success", "message": f"Action {action_name} onboarded", "action_id": action_id})
                print(f"[DB] Action {action_name} Onboarded with ID: {action_id}")
            elif operation == "update":
                action_data = {"action_name": action_name, "model_available": model_available, "tool": tool, "object": object}
                await self.action_db.update_action(action_id, action_data)
                response.update({"status": "success", "message": f"Successfully updated the action {action_name}"})
                print(f"[DB] Successfully updated the action {action_name}")
            elif operation == "delete":
                await self.action_db.delete_action(action_id)
                print(f"[DB] Successfully deleted the action {action_id}")
                response.update({"status": "success", "message": f"Successfully deleted the action {action_id}"})
                print(f"[DB] Successfully deleted the action {action_id}")
            elif operation == "read":
                data = await self.action_db.get_action(action_id)
                print(f"[DB] Retrieved action with ID: {action_id}")
                response.update({"status": "success", "message": "Action retrieved", "data": data})
            elif operation == "list":
                data = await self.action_db.list_actions()
                print(f"[DB] Successfully listed the actions")
                response.update({"status": "success", "message": "Actions listed", "data": data})
            else:
                response.update({"status": "failed", "message": f"Unsupported operation: {operation}"})
        except Exception as e:
            response.update({"status": "failed", "message": str(e)})
        
        return response

    async def grasp_orchestrator(self, operation: str, grasp_id: Optional[str] = None, object_name: Optional[str] = None, 
                      grasp_distance: Optional[float] = None, pickup_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        This function acts as an orchestrator for the grasp database functions.
        
        Args:
            operation (str): A variable to define which CRUD operation to perform in the database.
        
        Returns:
            dict: A standardized response JSON.
        """
        if not self.grasp_db:
            raise Exception("GraspDB is not initialized")
        
        response = {"operation": operation,"status": "failed","message": "","grasp_id": grasp_id,"object_name": object_name,"grasp_distance": grasp_distance,"pickup_mode": pickup_mode,"data": None}
        
        try:
            if operation == "create":
                grasp_data = {"object_name": object_name, "grasp_distance": grasp_distance, "pickup_mode": pickup_mode}
                grasp_id = await self.grasp_db.create_grasp(grasp_data)
                response.update({"status": "success", "message": f"Grasp for {object_name} onboarded", "grasp_id": grasp_id})
                print(f"[DB] Grasp for {object_name} Onboarded with ID: {grasp_id}")
            elif operation == "update":
                grasp_data = {"object_name": object_name, "grasp_distance": grasp_distance, "pickup_mode": pickup_mode}
                await self.grasp_db.update_grasp(grasp_id, grasp_data)
                response.update({"status": "success", "message": f"Successfully updated the grasp for {object_name}"})
                print(f"[DB] Successfully updated the grasp for {object_name}")
            elif operation == "delete":
                await self.grasp_db.delete_grasp(grasp_id)
                print(f"[DB] Successfully deleted the grasp {grasp_id}")
                response.update({"status": "success", "message": f"Successfully deleted the grasp {grasp_id}"})
                print(f"[DB] Successfully deleted the grasp {grasp_id}")
            elif operation == "read":
                data = await self.grasp_db.get_grasp(grasp_id)
                print(f"[DB] Retrieved grasp with ID: {grasp_id}")
                response.update({"status": "success", "message": "Grasp retrieved", "data": data})
            elif operation == "list":
                data = await self.grasp_db.list_grasps()
                print(f"[DB] Successfully listed the grasps")
                response.update({"status": "success", "message": "Grasps listed", "data": data})
            else:
                response.update({"status": "failed", "message": f"Unsupported operation: {operation}"})
        except Exception as e:
            response.update({"status": "failed", "message": str(e)})
        
        return response