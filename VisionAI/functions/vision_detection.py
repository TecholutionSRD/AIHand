"""
This file contains the codes for Gemini Model Inference.
"""
import os
import sys
import json
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import json_repair
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("[GEMINI] GEMINI_API_KEY is missing. Check your .env file.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from VisionAI.functions.utils import *
from Config.config import load_config


class GeminiInference:
    """
    Gemini Inference is the class used for various inference tasks using Google's Gemini Model.
    Mainly the task is of object detection.
    
    Args:
        api_key (str): API key for accessing the Gemini model.
        recording_dir (Path): Directory containing the video recording.
        inference_mode (bool): Flag to control whether inference should be performed.
        target_classes (List[str]): List of target classes for detection.
    """
    def __init__(self, config_path: Path, inference_mode: bool = False):
        config = load_config(config_path)
        self.config = config.get("Gemini", {})
        self.configure_gemini(gemini_api_key)

        self.model = genai.GenerativeModel(self.config.get("model_name", "gemini-1.5-flash-002"))
        self.recording_dir = Path(self.config.get("recording_dir", "./recordings"))
        self.inference_mode = inference_mode

        self.lock = threading.Lock()
        self.process_results = None
        self.target_classes = []

        # Prompts
        self.detection_prompt = (
            "You are a world-class vision AI. Identify objects with specific details (e.g., 'red soda can').\n"
            "Return JSON:\n{'<detailed_object_name>': [xmin, ymin, xmax, ymax]}\n"
            "Use numbering for multiple instances (e.g., 'red soda can_1')."
        )

    @staticmethod
    def configure_gemini(api_key: str) -> None:
        """Configures the Gemini API."""
        if not api_key:
            raise ValueError("[GEMINI] GEMINI_API_KEY is missing.")
        genai.configure(api_key=api_key)

    def set_target_classes(self, target_classes: List[str]) -> None:
        """Sets the target classes for detection."""
        self.target_classes = target_classes

    def process_frame(self, image: Image.Image) -> None:
        """
        Processes a given image frame using a pre-trained model to generate content and updates the detection results.
        Args:
            image (Image.Image): The image frame to be processed.
        Returns:
            None
        Raises:
            json.JSONDecodeError: If there is an error decoding the JSON response from the model.
            Exception: For any other exceptions that occur during processing.
        Notes:
            - The method uses a lock to ensure thread-safe updates to the process_results attribute.
            - If the model's response is empty or invalid, the process_results attribute is set to an empty dictionary.
        """

        try:
            response = self.model.generate_content([image, self.detection_prompt])

            if not response or not response.text:
                print("[GEMINI] Empty response from the model.")
                with self.lock:
                    self.process_results = {}
                return

            detection_results = json.loads(json_repair.repair_json(response.text))

        except (json.JSONDecodeError, Exception) as e:
            print(f"[GEMINI] Error: {e}")
            detection_results = {}

        with self.lock:
            self.process_results = detection_results

    def get_process_results(self) -> Optional[Dict]:
        """Returns the processed frame results."""
        with self.lock:
            return self.process_results

    def detect(self, image: Image.Image) -> List[str]:
        """
        Detects objects in the given image.
        Args:
            image (Image.Image): The image in which to detect objects.
        Returns:
            List[str]: A list of detected object names. If no objects are detected, returns an empty list.
        """
        
        self.process_frame(image)
        results = self.get_process_results()

        if not results:
            return []

        object_names = list(set(key.rsplit("_", 1)[0] for key in results.keys()))
        print(f"[GEMINI] Detected objects: {object_names}")
        return object_names

    def get_object_centers(self, image: Image.Image, target_objects: List[str], depth_frame: np.ndarray, intrinsics) -> List[List[float]]:
        """
        Get the real-world centers of specified target objects in an image.
        
        Args:
            image (Image.Image): The input image containing the objects.
            target_objects (List[str]): A list of target object names to find in the image.
            depth_frame (np.ndarray): The depth frame corresponding to the image.
            intrinsics: The camera intrinsics used for deprojection.
            
        Returns:
            List[List[float]]: A list of real-world coordinates [x, y, z] in robot base frame for the centers of the target objects.
        """
        if self.process_results is None:
            self.process_frame(image)

        results = self.get_process_results()
        if not results:
            return []

        object_centers = []

        for obj in target_objects:
            matching_keys = [key for key in results.keys() if key.startswith(obj)]

            if not matching_keys:
                print(f"[GEMINI] No matching object found for: {obj}")
                continue

            for key in matching_keys:
                try:
                    box = self.normalize_box(results[key])
                    center_x = int((box[0] + box[2]) // 2)
                    center_y = int((box[1] + box[3]) // 2)

                    camera_coords = deproject_pixel_to_point(depth_frame, (center_x, center_y), intrinsics)
                    transformed_coords = transform_coordinates(camera_coords[0], camera_coords[1], camera_coords[2])
                    object_centers.append(list(transformed_coords))

                except Exception as e:
                    print(f"[GEMINI] Error processing {obj}: {e}")

        return object_centers

    def normalize_box(self, box, width=640, height=480):
        """
        Normalize bounding boxes from pixel coordinates to [0, 1] range.

        Args:
            boxes (list): List of bounding boxes in [ymin, xmin, ymax, xmax] format.
            width (int): Image width.
            height (int): Image height.

        Returns:
            list: Normalized bounding boxes in [ymin, xmin, ymax, xmax] format.
        """
        ymin, xmin, ymax, xmax = box
        normalized_box = [ xmin / 1000*width, ymin / 1000*height, xmax / 1000*width, ymax / 1000*height]
        return normalized_box