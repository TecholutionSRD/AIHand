"""
This file contains the functions to check the database and manipulate it.
# TODO : For now the database is a csv file, but it should be a MongoDB database.
"""
import pandas as pd
import os
import sys
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config

config = load_config("db_config.yaml")
db_config = config.get("DataBase", {})

#--------------------------------------------------------------#
def ensure_csv_file_exists(file_path, columns):
    """Ensures a CSV file exists with given columns."""
    if not os.path.exists(file_path):
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)

def ensure_directory_exists(directory):
    """Ensures that the given directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
#--------------------------------------------------------------#
def check_knowledgebase(db_config):
    """
    Checks the existence and validity of the knowledge base files.
    This function ensures that the necessary CSV files for grasping data,
    action data, and action-object relationships exist in the specified
    directory. It creates the base directory if it doesn't exist and
    initializes the CSV files with headers if they are new. It also reads
    the CSV files into pandas DataFrames, converts the 'model' column in
    the action data to boolean type, and extracts lists of grasp names and
    action names.
    Args:
        db_config (dict): A dictionary containing the configuration for the
            database. It should have the following keys:
            - "base_dir" (str, optional): The base directory for the database
                files. Defaults to "./database".
            - "grasp" (str, optional): The filename for the grasping data CSV.
                Defaults to "grasping_data.csv".
            - "action" (str, optional): The filename for the action data CSV.
                Defaults to "action_data.csv".
            - "action_objects" (str, optional): The filename for the
                action-objects CSV. Defaults to "action_objects.csv".
    Returns:
        tuple: A tuple containing two lists:
            - grasp_names (list): A list of grasp names from the grasping data.
            - action_names (list): A list of action names from the action data
                where the 'model' column is True.
    """
    base_dir = Path(db_config.get("base_dir", "./Database"))
    grasp_db_path = base_dir / db_config.get("grasp", "grasp.csv")
    action_db_path = base_dir / db_config.get("action", "action.csv")
    action_objects_db_path = base_dir / db_config.get("action_objects", "action_objects.csv")

    base_dir.mkdir(parents=True, exist_ok=True)

    ensure_csv_file_exists(grasp_db_path, ["name", "grasp_distance", "pickup_mode"])
    ensure_csv_file_exists(action_db_path, ["id", "name", "samples", "model"])
    ensure_csv_file_exists(action_objects_db_path, ["action_id", "object_name"])

    grasp_df = pd.read_csv(grasp_db_path)
    action_df = pd.read_csv(action_db_path)
    
    # Ensure model column is boolean
    if "model" in action_df.columns:
        action_df["model"] = action_df["model"].astype(bool)

    grasp_names = grasp_df["name"].tolist()
    action_names = action_df[action_df["model"]]["name"].tolist()

    print(f"[Database] Base Directory: {base_dir}")
    print(f"[Database] Grasping Data: {grasp_db_path}")
    print(f"[Database] Action Data: {action_db_path}")
    print(f"[Database] Action-Objects Data: {action_objects_db_path}")
    return grasp_names, action_names

#--------------------------------------------------------------#
def specific_knowledge_check(db_config, name):
    """
    Checks for the existence of a specific object's knowledge in the database.
    Args:
        db_config (dict): A dictionary containing the database configuration, including:
            - base_dir (str): The base directory where the database files are located.
            - grasp (str): The filename of the grasp database CSV.
            - action (str): The filename of the action database CSV.
            - action_objects (str, optional): The filename of the action objects database CSV. Defaults to "action_objects.csv".
        name (str): The name of the object to check for.
    Returns:
        dict: A dictionary containing the results of the checks:
            - name (str): The name of the object.
            - grasp (bool): True if the object exists in the grasp database, False otherwise.
            - action (bool): True if the object exists in the action database and its 'model' column is True, False otherwise.
            - object_in_action (bool): True if the object exists in the action_objects database, False otherwise.
            - error (str, optional): An error message if the database configuration is invalid.
    """
    base_dir = db_config.get("base_dir")
    grasp_db = db_config.get("grasp")
    action_db = db_config.get("action")
    action_objects_db = db_config.get("action_objects", "action_objects.csv")
    
    if not all([base_dir, grasp_db, action_db]):
        print(f'[Database] Base Dir : {base_dir}')
        print(f'[Database] Grasp Dir : {grasp_db}')
        print(f'[Database] Action Dir : {action_db}')
        print("[Database] Missing Database files.")
        return {"error": "Invalid database configuration"}

    grasp_db_path = os.path.join(base_dir, grasp_db)
    action_db_path = os.path.join(base_dir, action_db)
    action_objects_db_path = os.path.join(base_dir, action_objects_db)

    if not (os.path.exists(grasp_db_path) and os.path.exists(action_db_path)):
        return {"name": name, "grasp": False, "action": False, "object_in_action": False}

    try:
        grasp_df = pd.read_csv(grasp_db_path) if os.path.getsize(grasp_db_path) > 0 else pd.DataFrame(columns=["name"])
        action_df = pd.read_csv(action_db_path) if os.path.getsize(action_db_path) > 0 else pd.DataFrame(columns=["name", "model"])
    except Exception as e:
        print(f"[Database] Error reading CSV files: {e}")
        return {"name": name, "grasp": False, "action": False, "object_in_action": False}

    grasp_exists = name in grasp_df["name"].tolist()
    action_exists = name in action_df[action_df["model"].astype(bool)]["name"].tolist()

    object_in_action = False
    if os.path.exists(action_objects_db_path):
        try:
            action_objects_df = pd.read_csv(action_objects_db_path)
            object_in_action = name in action_objects_df["object_name"].tolist()
        except Exception as e:
            print(f"[Database] Error reading action_objects CSV: {e}")

    return {"name": name, "grasp": grasp_exists, "action": action_exists, "object_in_action": object_in_action}


def get_action_objects(db_config, action_name):
    """
    Retrieves a list of object names associated with a specific action from CSV files.
    Args:
        db_config (dict): A dictionary containing database configuration parameters,
                            including 'base_dir', 'action' (path to action database CSV),
                            and 'action_objects' (path to action objects database CSV).
        action_name (str): The name of the action to retrieve objects for.
    Returns:
        list: A list of object names (strings) associated with the specified action.
                Returns an empty list if the action is not found, if the database
                files do not exist, or if the configuration is incomplete.
    """
    
    base_dir = db_config.get("base_dir")
    action_db = db_config.get("action")
    action_objects_db = db_config.get("action_objects", "action_objects.csv")

    if not base_dir or not action_db:
        return []

    action_db_path = os.path.join(base_dir, action_db)
    action_objects_db_path = os.path.join(base_dir, action_objects_db)

    if not os.path.exists(action_db_path) or not os.path.exists(action_objects_db_path):
        return []

    action_df = pd.read_csv(action_db_path)
    action_objects_df = pd.read_csv(action_objects_db_path)

    action_row = action_df[action_df["name"] == action_name]
    if action_row.empty:
        return []

    action_id = action_row.iloc[0]["id"]
    return action_objects_df[action_objects_df["action_id"] == action_id]["object_name"].tolist()

def add_grasp_data(db_config, name, grasp_distance, pickup_mode):
    """
    Adds grasp data to the grasp database.
    
    Args:
        db_config (dict): A dictionary containing database configuration parameters.
        name (str): The name of the object/grasp.
        grasp_distance (float): The grasp distance value.
        pickup_mode (str): The pickup mode used for the grasp.
    
    Returns:
        tuple: (bool, str) - True and success message if grasp data was added, 
                             False and a message otherwise.
    """

    base_dir = db_config.get("base_dir")
    grasp_db = db_config.get("grasp")

    if not base_dir or not grasp_db:
        return False, "[Database] Missing 'base_dir' or 'grasp' database filename in configuration."

    ensure_directory_exists(base_dir)
    grasp_db_path = os.path.join(base_dir, grasp_db)
    ensure_csv_file_exists(grasp_db_path, ["name", "grasp_distance", "pickup_mode"])

    try:
        grasp_df = pd.read_csv(grasp_db_path)

        # Ensure the CSV has the required columns
        if set(["name", "grasp_distance", "pickup_mode"]).issubset(grasp_df.columns):
            if name in grasp_df["name"].tolist():
                return False, f"[Database] Grasp entry for '{name}' already exists."

        else:
            return False, "[Database] CSV file is missing required columns."

        # Add new entry
        new_row = pd.DataFrame([[name, grasp_distance, pickup_mode]], 
                                columns=["name", "grasp_distance", "pickup_mode"])
        grasp_df = pd.concat([grasp_df, new_row], ignore_index=True)
        grasp_df.to_csv(grasp_db_path, index=False)

        return True, f"[Database] Successfully added grasp entry for '{name}'."

    except Exception as e:
        return False, f"[Database] Error while adding grasp entry: {e}"

def add_action_data(db_config, name, samples, model=True):
    """
    Adds action data to the action database.
    
    Args:
        db_config (dict): A dictionary containing the database configuration.
        name (str): The name of the action to add.
        samples (int): The number of samples associated with the action.
        model (bool, optional): A boolean indicating whether a model is
                                associated with the action. Defaults to True.
    
    Returns:
        int: The ID of the newly added action, or the existing ID if the
             action already exists. Returns None if the database configuration
             is incomplete.
    """

    base_dir = db_config.get("base_dir")
    action_db = db_config.get("action")

    if not base_dir or not action_db:
        return None

    ensure_directory_exists(base_dir)
    action_db_path = os.path.join(base_dir, action_db)
    ensure_csv_file_exists(action_db_path, ["id", "name", "samples", "model"])

    action_df = pd.read_csv(action_db_path)
    print("[Debug] Database Loaded")

    if name in action_df["name"].tolist():
        action_id = action_df[action_df["name"] == name].iloc[0]["id"]
        print(f"[Database] Action entry for '{name}' already exists with ID {action_id}.")
        return int(action_id) 
    
    new_id = 1 if action_df.empty else int(action_df["id"].max() + 1)
    new_row = pd.DataFrame([[new_id, name, samples, model]], 
                            columns=["id", "name", "samples", "model"])
    
    action_df = pd.concat([action_df, new_row], ignore_index=True)
    action_df.to_csv(action_db_path, index=False)

    print(f"[Database] Added action entry for '{name}' with ID {new_id}.")
    return new_id

def add_action_object(db_config, action_id, object_name):
    """
    Associates an object with an action in the action_objects database.
    
    Args:
        db_config (dict): A dictionary containing database configuration parameters.
        action_id (int): The ID of the action to associate with the object.
        object_name (str): The name of the object to associate with the action.
    
    Returns:
        dict: A response dictionary with success status and message.
    """
    base_dir = db_config.get("base_dir")
    action_objects_db = db_config.get("action_objects", "action_objects.csv")

    if not base_dir:
        return {"success": False, "error": "Database configuration missing"}

    ensure_directory_exists(base_dir)
    action_objects_db_path = os.path.join(base_dir, action_objects_db)
    ensure_csv_file_exists(action_objects_db_path, ["action_id", "object_name"])

    action_objects_df = pd.read_csv(action_objects_db_path)

    # Convert action_id to int to avoid numpy.int64 issues
    action_id = int(action_id)

    # Check if association already exists
    if ((action_objects_df["action_id"] == action_id) & 
        (action_objects_df["object_name"] == object_name)).any():
        print(f"[Database] Association between action ID {action_id} and object '{object_name}' already exists.")
        return {"success": False, "error": "Association already exists"}

    # Add new association
    new_row = pd.DataFrame([[action_id, object_name]], columns=["action_id", "object_name"])
    action_objects_df = pd.concat([action_objects_df, new_row], ignore_index=True)
    action_objects_df.to_csv(action_objects_db_path, index=False)

    print(f"[Database] Added association between action ID {action_id} and object '{object_name}'.")
    return {"success": True, "message": "Association added successfully"}


