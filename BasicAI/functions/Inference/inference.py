"""
Author: Shreyas Dixit
This file contains the inference class for the base model.
"""
from pathlib import Path
from typing import List
import joblib
import pandas as pd
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from BasicAI.functions.trainer.model import TrajectoryModel
from Camera.functions.camera_receiver import CameraReceiver
from VisionAI.functions.vision_detection import GeminiInference
from VisionAI.functions.llm_router import *
from Config.config import load_config


class TrajectoryInference:
    """Class for performing inference with the trained Trajectory Prediction model."""

    def __init__(self, config_path: str = "basic_ai_config.yaml", checkpoint_dir: str = "checkpoints"):
        """
        Initialize the inference class by loading model, scalers, and configurations.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            checkpoint_dir (str): Directory containing model checkpoints and scalers.
        """
        print("[Inference] Initializing inference module...")

        config = load_config(config_path)['Architecture']['NeuralNet']
        self.input_shape = config['input_shape']
        self.output_shape = config['output_shape']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_path = self.checkpoint_dir / "best_model.pth"
        self.input_scaler_path = self.checkpoint_dir / "input_scaler.pkl"
        self.output_scaler_path = self.checkpoint_dir / "output_scaler.pkl"

        self._check_files_exist()

        self.model = self._load_model()
        self.input_scaler, self.output_scaler = self._load_scalers()

    def _check_files_exist(self):
        """Ensure required files (model & scalers) exist before proceeding."""
        missing_files = [f for f in [self.model_path, self.input_scaler_path, self.output_scaler_path] if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"[Inference] ERROR: Missing required files: {missing_files}")

    def _load_model(self) -> TrajectoryModel:
        """Load the trained model from checkpoint."""
        print(f"[Inference] Loading model from {self.model_path}...")
        model = TrajectoryModel(self.input_shape, self.output_shape).to(self.device)
        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            print("[Inference] Model loaded successfully.")
        except RuntimeError as e:
            print(f"[Inference] Model weights do not match. Please check the configs {e}")
            raise
        return model

    def _load_scalers(self):
        """Load input and output scalers."""
        print("[Inference] Loading scalers...")
        input_scaler = joblib.load(self.input_scaler_path)
        output_scaler = joblib.load(self.output_scaler_path)
        print("[Inference] Scalers loaded successfully.")
        return input_scaler, output_scaler

    def predict(self, points: list) -> list:
        """
        Perform inference using the trained model.

        Args:
            points (list): List of two 3D coordinates (x1, y1, z1, x2, y2, z2).

        Returns:
            list: Predicted trajectory points in (X, Y, Z, C) format.
        """
        print("[Inference] Preprocessing input data...")

        points_scaled = self.input_scaler.transform([sum(points, [])])
        points_tensor = torch.tensor(points_scaled, dtype=torch.float32).to(self.device)

        print("[Inference] Running model prediction...")
        with torch.no_grad():
            output = self.model(points_tensor)
            output = output.cpu().numpy()

        print("[Inference] Transforming output back to original scale...")
        output = self.output_scaler.inverse_transform(output).reshape(-1, 4).tolist()

        return output

    def save_output(self, output: list, output_path: Path = Path("output/output.csv")):
        """
        Save the predicted output to a CSV file.

        Args:
            output (list): List of predicted trajectory points.
            output_path (Path): Path where the output CSV should be saved.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as f:
            f.write('X,Y,Z,C\n')
            for line in output:
                f.write(','.join(map(str, line)) + '\n')

    def run_inference(self, points: list) -> Path:
        """
        Perform inference and save the output.

        Args:
            points (list): List of 6 values representing two points.

        Returns:
            Path: Path to the output CSV file.
        """
        print("[Inference] Starting inference process...")
        predictions = self.predict(points)
        output_path = Path("output/output.csv")
        self.save_output(predictions, output_path)
        print("[Inference] Process complete.")
        return output_path


# async def realtime_inference(camera: CameraReceiver, gemini: GeminiInference, target_classes: List[str]):
#     """
#     Main inference pipeline: Captures an image, runs object detection, and predicts trajectory.

#     Args:
#         camera: CameraReceiver object for capturing images
#         gemini: GeminiInference object for object detection
#         target_classes: List of target class names to detect
#     """
#     #TODO: Add Infernce model.
#     trajectory_model = None
#     real_world_centers = []

#     frames = await camera.capture_frame()
#     color_frame_path = frames.get("rgb")
#     depth_frame_path = frames.get("depth")

#     if not color_frame_path or not depth_frame_path:
#         print("[Error] Failed to capture images.")
#         return None

#     intrinsics = camera._get_intrinsics(location="India", camera_name="D435I")

#     print(f"[Inference] Processing detection for {len(target_classes)} target classes: {target_classes}")
#     for class_name in target_classes:
#         print(f"[Inference] Detecting {class_name}...")

#         transformed_center = await gemini.detect(camera, color_frame_path, depth_frame_path, class_name, intrinsics)

#         real_world_centers.append(transformed_center)

#     if all(center is None for center in real_world_centers):
#         print("[Error] No valid detections found. Cannot proceed with trajectory prediction.")
#         return None

#     print("[Inference] Running trajectory prediction...")
#     print("-" * 100)
#     print(f"[Inference] {real_world_centers}")
#     print("-" * 100)
#     try:
#         output_path = trajectory_model.run_inference(real_world_centers)
#         print(f"[Inference] Trajectory prediction saved to: {output_path}")
#     except Exception as e:
#         print(f"[Inference] Error during trajectory prediction: {e}")
#         return None

#     print("[Inference] Inference pipeline completed successfully.")
#     return output_path

#-----------------------------------------------------------------------------------------#
def save_simulation_json(action_name, after_hamer_inputs, predictions, output_file="./output.json"):
    """
    Save the trajectory predictions and fundamental actions into a JSON file with a timestamped filename.
    """
    data = {
        "objects_to_spawn": {
            "plate": [coord / 1000 for coord in after_hamer_inputs[:3]],
            "bread": [coord / 1000 for coord in after_hamer_inputs[:3]],
            "ketchup": [coord / 1000 for coord in after_hamer_inputs[3:6]]
        },
        "fundamental_actions": {
            "grasp": {
                "is_trajectory": False,
                "coordinates": [coord / 1000 if i < 2 else (coord - 0) / 1000 for i, coord in enumerate(after_hamer_inputs[3:6])],
                "csv_data": "",
                "gripper_mode_post_action": 2,
                "grasp_mode": "HORIZONTAL"
            },
            action_name: {
                "is_trajectory": True,
                "coordinates": [0, 0, 0, 0, 0, 0],  
                "csv_data": predictions,
                "gripper_mode_post_action": 0,
                "grasp_mode": "HORIZONTAL"
            },
            "place": {
                "is_trajectory": False,
                "coordinates": [coord / 1000 for coord in after_hamer_inputs[3:6]],
                "csv_data": "",
                "gripper_mode_post_action": 1,
                "grasp_mode": "HORIZONTAL"
            }
            
        }
    }

    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"[INFO] JSON file saved: {output_file}")
    return output_file
#-----------------------------------------------------------------------------------------#
# This is an Inference function written for larger tasks this will later replace the above function.
async def inference(camera: CameraReceiver, gemini: GeminiInference, action_name: str = None, target_classes: List[str] = None):
    """
    Performs an inference operation by analyzing input from a camera, retrieving relevant actions and objects, 
    and interacting with various databases to determine the appropriate execution steps.

    This function integrates multiple API calls, including:
    - Capturing an image from a camera.
    - Processing the image to detect objects.
    - Using an LLM (Gemini) to match detected objects to known database objects.
    - Querying the Action Database (ActionDB) to retrieve corresponding action details.
    - Interacting with the Grasp Database (GraspDB) to determine object centers and grasping feasibility.
    - Performing inference using machine learning models.
    - Sending the processed information for further in simulation mode.

    The function follows this sequence:
    1. Captures an image from the `CameraReceiver`.
    2. Detects objects within the image and attempts to map them to `target_classes`.
    3. Queries the `ActionDB` to retrieve the action name, object names, and model paths.
    4. Matches user-defined objects to database objects using the Gemini LLM.
    5. Checks with `GraspDB` to get object centers.
    6. If objects are found:
       - Loads the appropriate model.
       - Performs inference to generate a payload.
       - Sends the payload to the execution system.
    7. If objects are not found, it prompts the user for additional input to "Teach Grasp."

    Parameters:
        camera (CameraReceiver): An interface for capturing images from the camera.
        gemini (GeminiInference): The LLM-based inference engine used to match objects.
        action_name (str, optional): The specific action to perform. Defaults to None.
        target_classes (List[str], optional): A list of target object classes to detect. Defaults to None.

    Returns:
        dict: A structured response containing:
              - `action_name`: The determined action name.
              - `objects_detected`: List of detected objects.
              - `inference_result`: The inference output.
              - `execution_payload`: The final payload sent for execution.

    Raises:
        ValueError: If no valid objects are detected.
        ConnectionError: If there is an issue connecting to external APIs (ActionDB, GraspDB).
        RuntimeError: If inference fails due to model loading or execution issues.

    Example Usage:
        ```python
        result = await inference(camera=my_camera, gemini=my_gemini, action_name="PickObject", target_classes=["bottle", "cup"])
        print(result)
        ```
    """
    #-------------------------------------------------------------------------#
    # Capture Image
    #-------------------------------------------------------------------------#
    intrinsics = camera._get_intrinsics(location="India", camera_name="D435I")
    frames = await camera.capture_frame()
    color_frame_path = frames.get("color_frame_path")
    depth_frame_path = frames.get("depth_frame_path")

    if not color_frame_path or not depth_frame_path:
        raise ValueError("[Error] Failed to capture images from camera.")

    #-------------------------------------------------------------------------#
    # Detect Objects
    #-------------------------------------------------------------------------#
    detected_objects = []
    if target_classes:
        for class_name in target_classes:
            print(f"[Inference] Detecting {class_name}...")
            transformed_center = await gemini.detect(camera, color_frame_path, depth_frame_path, class_name, intrinsics)
            if transformed_center:
                detected_objects.append({"class_name": class_name, "center": transformed_center})

    if not detected_objects:
        raise ValueError("[Error] No valid objects detected.")
    #-------------------------------------------------------------------------#
    # Validate Action and Grasp Database
    #-------------------------------------------------------------------------#
    print("[Inference] Validating task requirements...")
    object_names = [obj["class_name"] for obj in detected_objects]
    
    validation_result = await validate_task(action_name, object_names)
    if validation_result["status"] == "error":
        raise RuntimeError(f"[Error] Task validation failed: {validation_result['message']}")

    action_details = validation_result["action_details"]
    grasp_details = validation_result["grasp_details"]

    if not grasp_details:
        raise RuntimeError("[Error] No grasp details found for detected objects")

    print(f"[Inference] Task validation successful: {validation_result['message']}")

    validation_details = {
        "action_name": action_name,
        "objects_detected": detected_objects,
        "action_details": action_details,
        "grasp_details": grasp_details,
        "status": "success"
    }
    
    #-------------------------------------------------------------------------#
    # Update object centers with grasp distance and run trajectory inference
    #-------------------------------------------------------------------------#
    print("[Inference] Preparing object centers for trajectory prediction...")
    
    trajectory_model = TrajectoryInference()
    object_centers = []
    
    # Extract tool and target object centers and update with grasp distances
    for obj in detected_objects:
        center = obj["center"]
        obj_name = obj["class_name"]
        grasp_distance = grasp_details.get(obj_name, {}).get("grasp_distance", 0)
        
        # Add grasp distance to Z coordinate
        updated_center = [center[0], center[1], center[2] + grasp_distance]
        object_centers.append(updated_center)

    if len(object_centers) != 2:
        raise ValueError("[Error] Exactly two objects (tool and target) required for trajectory prediction")

    try:
        output_path = trajectory_model.run_inference(object_centers)
        print(f"[Inference] Trajectory prediction saved to: {output_path}")
        csv_data = pd.read_csv(output_path)
        sim_data = csv_data.to_csv(index=False)

    except Exception as e:
        raise RuntimeError(f"[Error] Trajectory prediction failed: {str(e)}")
    
    #-------------------------------------------------------------------------#
    # Data to Simulation
    #-------------------------------------------------------------------------#
    # Prepare inputs for JSON
    after_hamer_inputs = []
    for center in object_centers:
        after_hamer_inputs.extend(center)  # Flatten the centers into a single list

    try:
        # Save trajectory data to JSON
        output_json = save_simulation_json(action_name=action_name,after_hamer_inputs=after_hamer_inputs,predictions=sim_data)
        print(f"[Inference] Successfully saved trajectory data to {output_json}")
        with open(output_path, 'r') as f:
                payload = json.load(f)
        
        return {
            "message": "Inference completed successfully",
            "payload": payload,
            "validation_details": validation_details,
            "status": "success"
        }
        
    except Exception as e:
        raise RuntimeError(f"[Error] Failed to save trajectory data: {str(e)}")
#-----------------------------------------------------------------------------------------#