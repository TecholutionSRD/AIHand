"""
This file contains the codes for Google's Gemini Model Inference.
"""

import asyncio
import time
import os
import threading
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import json_repair
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import sys

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from VisionAI.utils.utils import *

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
    def __init__(self, config, inference_mode: bool = False):
        self.config = config.get("Gemini", {})
        self.configure_gemini(gemini_api_key)
        self.model = genai.GenerativeModel(model_name=self.config["model_name"])
        self.recording_dir = Path(self.config["recording_dir"])
        self.inference_mode = inference_mode

        self.lock = threading.Lock()
        self.detection_results = None
        self.process_results = None
        self.boxes = None
        self.target_classes = []

        self.default_prompt = (
            "Return bounding boxes for objects in the format:\n"
            "```json\n{'<object_name>': [xmin, ymin, xmax, ymax]}\n```\n"
            "Include multiple instances as '<object_name>_1', '<object_name>_2', etc."
        )
        self.detection_prompt = (
            "You are a world-class computer vision expert. Analyze this image carefully and detect "
            "all objects with detailed, specific descriptions. "
            "Return bounding boxes in JSON format:\n"
            "{'<detailed_object_name>': [xmin, ymin, xmax, ymax]}\n"
            "Use '<detailed_object_name>_1', '<detailed_object_name>_2' for multiple instances."
        )
        self.capture_state = False

    @staticmethod
    def configure_gemini(api_key: str) -> None:
        """Configure Gemini API."""
        genai.configure(api_key=api_key)

    def set_target_classes(self, target_classes: List[str]) -> None:
        """
        Set the target classes for detection.

        Args:
            target_classes (List[str]): List of target classes.
        """
        self.target_classes = target_classes

    def process_frame(self, image: Image.Image):
        """
        Process a single frame for object detection.

        Args:
            image (Image.Image): The input image.
        """
        prompt = self.default_prompt
        if self.target_classes:
            prompt += "\nDetect the following classes: " + ", ".join(self.target_classes)

        response = self.model.generate_content([image, prompt])
        try:
            detection_results = json.loads(json_repair.repair_json(response.text))
        except ValueError as e:
            detection_results = {}
            print(f"[Detect] Error parsing detection results: {e}")

        with self.lock:
            self.process_results = detection_results

    def get_process_frame_results(self) -> Optional[Dict]:
        """Returns processed frame results."""
        with self.lock:
            return self.process_results

    def set_inference_state(self, state: bool):
        """Enable or disable inference mode."""
        self.inference_mode = state

    def set_capture_state(self, state: bool):
        """Enable or disable capture mode."""
        self.capture_state = state

    async def get_object_center(self, camera, target_class: str) -> Optional[Dict]:
        """
        Get the center and bounding box of a detected object.

        Args:
            image (Image.Image): The input image.
            target_class (str): The target class name.

        Returns:
            dict: Contains center coordinates, bounding box, and confidence score.
        """
        frames = await camera.capture_frame()
        color_frame_path = frames.get("rgb")

        image = Image.open(color_frame_path)
        self.process_frame(image)
        results = self.get_process_frame_results()

        print(f"[Detect] Processing results: {results}")

        if not results or target_class not in results:
            return None

        box = self.normalize_box(results[target_class])
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        detection_results = {"center": (center_x, center_y), "box": box, "confidence": 100}

        print(f"[Detect] Normalized Box: {box}")
        self.detection_results = detection_results
        print(f"[Detect] Detection Results", self.detection_results)
        return self.detection_results

    async def detect(self, camera, target_class: List[str] = ["red soda can"]):
        """
        Processes images to detect objects and calculate real-world coordinates.

        Args:
            camera: Camera instance.
            target_class (List[str]): List of target objects to detect.

        Returns:
            Transformed real-world coordinates of the detected object.
        """
        recording_dir = self.config.get("recording_dir")
        frames = await camera.capture_frame()
        color_frame_path = frames.get("rgb")
        depth_frame_path = frames.get("depth")

        intrinsics = camera._get_intrinsics(location="India", camera_name="D435I")

        self.set_target_classes(target_class)
        color_image = Image.open(color_frame_path)

        print("[Detect] Processing frame for object detection...")
        output = self.get_object_center(color_image, target_class[0])
        print(f"[Detect] Detection Output: {output}")

        pixel_center = output.get("center") if output else None
        if not pixel_center:
            print("[Detect] No object detected.")
            return None

        depth_image = np.load(depth_frame_path)
        try:
            depth_center = deproject_pixel_to_point(depth_image, pixel_center, intrinsics=intrinsics)
        except Exception as e:
            print(f"[Detect] Error deprojecting pixel: {e}")
            return None

        transformed_center = transform_coordinates(*depth_center)
        print(f"[Detect] Transformed Center: {transformed_center}")

        return transformed_center
    
    async def detect_all(self, camera):
        """
        Processes images to detect objects and calculate real-world coordinates.

        Args:
            camera: Camera instance.
            target_class (List[str]): List of target objects to detect.

        Returns:
            Transformed real-world coordinates of the detected object.
        """
        frames = await camera.capture_frame()
        color_frame_path = frames.get("rgb")

        color_image = Image.open(color_frame_path)

        print("[Detect] Processing frame for object detection...")
        output = self.detect_objects(color_image)
        print(f"[Detect] Detection Output: {output}")
        return output

    def detect_objects(self, rgb_frame: Image.Image) -> List[str]:
        """
        Run detection on a single RGB frame and return detected object names.

        Args:
            rgb_frame (Image.Image): RGB frame as PIL Image.

        Returns:
            List[str]: List of detected object names.
        """
        response = self.model.generate_content([rgb_frame, self.detection_prompt])
        try:
            results = json.loads(json_repair.repair_json(response.text))
        except ValueError as e:
            print(f"[Detect] Error parsing detection results: {e}")
            return []

        with self.lock:
            self.process_results = results

        object_names = list(set(key.rsplit("_", 1)[0] for key in results.keys()))
        print(f"[Detect] Detected objects: {object_names}")

        return object_names

    def normalize_box(self, box, width=640, height=480):
        """
        Normalize bounding boxes from pixel coordinates to [0, 1] range.

        Args:
            box (list): Bounding box in [ymin, xmin, ymax, xmax] format.

        Returns:
            list: Normalized bounding boxes.
        """
        ymin, xmin, ymax, xmax = box
        return [xmin / 1000 * width, ymin / 1000 * height, xmax / 1000 * width, ymax / 1000 * height]


# if __name__ == "__main__":
#     async def main():
#         config = load_config("config/config.yaml")
#         camera = CameraReceiver(config)
#         gemini = GeminiInference(config)
#         detected_objects = await gemini.detect(camera, target_class=["bottle"])
#         print(f"[Detect] Final Detected Objects: {detected_objects}")

#     asyncio.run(main())
