"""
This module manages task operations for the RLEF system.
"""

import os
import requests
from typing import Optional

from config.config import load_config

class RLEFManager:
    """
    A class to manage task operations for the RLEF.
    """
    def __init__(self, config: dict):
        """Initialize RLEFManager instance with config values."""
        self.BASE_URL = config.get("base_url")
        self.url = config.get("url")
        self.token = config.get("token")
        self.model_group_id = config.get("model_group_id") 
        self.project_id = config.get("project_id")
        self.task_type = config.get("task_type", "videoAnnotation")

        if not self.token or not self.model_group_id or not self.project_id:
            raise ValueError("Missing required authentication details. Ensure environment variables are set.")

    def _get_headers(self) -> dict:
        """Returns headers for API requests."""
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def fetch_tasks(self, task_name: Optional[str] = None) -> Optional[str]:
        """
        Fetch task ID by task name.

        Args:
            task_name (Optional[str]): Name of the task.

        Returns:
            Optional[str]: Task ID if found, None otherwise.
        """
        params = {"modelGroupId": self.model_group_id}
        if task_name:
            params["name"] = task_name

        try:
            response = requests.get(self.BASE_URL, headers=self._get_headers(), params=params)
            response.raise_for_status()
            data = response.json()
            return data["models"][0]["_id"] if data.get("models") else None
        except requests.exceptions.RequestException as e:
            print(f"[RLEF] Error fetching tasks: {e}")
            return None

    def create_task(self, task_name: str) -> Optional[str]:
        """
        Create a new task.

        Args:
            task_name (str): Name of the task.

        Returns:
            Optional[str]: ID of the created task, None if failed.
        """
        payload = {
            "name": task_name,
            "type": self.task_type,
            "modelGroupId": self.model_group_id,
            "project": self.project_id
        }

        try:
            response = requests.post(self.BASE_URL, headers=self._get_headers(), json=payload)
            response.raise_for_status()
            return response.json().get("_id")
        except requests.exceptions.RequestException as e:
            print(f"[RLEF] Error creating task: {e}")
            return None

    def get_or_create_task(self, task_name: str) -> Optional[str]:
        """
        Fetch an existing task or create a new one.

        Args:
            task_name (str): Name of the task.

        Returns:
            Optional[str]: Task ID.
        """
        task_id = self.fetch_tasks(task_name)
        if task_id:
            print(f"[RLEF] Task '{task_name}' already exists with ID: {task_id}")
            return task_id

        print(f"[RLEF] Creating new task: '{task_name}'")
        return self.create_task(task_name)

    @staticmethod
    def convert_video(input_path: str, output_path: str) -> None:
        """
        Convert a video to the required format.

        Args:
            input_path (str): Path to the input video file.
            output_path (str): Path to save the converted video.
        """
        os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{output_path}'")

    def upload_to_rlef(self, filepath: str, task_id: Optional[str] = None) -> int:
        """
        Uploads video file to RLEF.

        Args:
            filepath (str): Path to the video file.
            task_id (Optional[str]): Task ID. Defaults to a predefined task.

        Returns:
            int: HTTP response status code.
        """
        if not task_id:
            print("[RLEF] No Task ID provided. Using default Task ID.")
            task_id = "67695dc462913593227a4227"

        converted_filepath = f"{filepath}_converted.mp4"
        self.convert_video(filepath, converted_filepath)

        payload = {
            "model": task_id,
            "status": "backlog",
            "csv": "csv",
            "label": "objects",
            "tag": "boxes",
            "prediction": "predicted",
            "confidence_score": "100",
            "videoAnnotations": {},
        }

        with open(converted_filepath, "rb") as file:
            files = {"resource": (converted_filepath, file)}

            try:
                response = requests.post(self.url, headers=self._get_headers(), data=payload, files=files)
                response.raise_for_status()
                print(f"[RLEF] Upload successful: {response.text}")
                return response.status_code
            except requests.exceptions.RequestException as e:
                print(f"[RLEF] Upload failed: {e}")
                return 500

async def rlef_upload(video_paths,action_name):
    try:
        rlef_config = load_config("Database/config/config.yaml")['RLEF']
        manager = RLEFManager(rlef_config)
        task_id = manager.get_or_create_task(action_name)
        print(f"[RLEF] Current_Task_ID: {task_id}")
        manager.upload_to_rlef(video_paths[0], task_id)
        print(f"[RLEF] Upload completed successfully")
        return True
    except Exception as e:
        print(f"[RLEF] Upload failed: {e}")
        return False

if __name__ == "__main__":
    config = load_config("../config/config.yaml").get("RLEF", {})
    
    try:
        manager = RLEFManager(config)
        
        task_name = "Pouring"
        task_id = manager.get_or_create_task(task_name)
        
        filepath = "data/recordings/pouring/sample_1/pouring_video.mp4"
        status_code = manager.upload_to_rlef(filepath, task_id)
        
        print(f"[RLEF] Upload completed with status code: {status_code}")

    except Exception as e:
        print(f"[RLEF] Unexpected error: {e}")
