import os
import requests
import sys
import uuid
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Config.config import load_config
"""
This file contains RLEF utility functions.
"""
class RLEFManager:
    """
    A class to manage task operations for the RLEF system.
    
    Attributes:
        BASE_URL (str): The base URL for the AutoAI backend API.
    """
    def __init__(self,config):
        """Initialize TaskManager instance."""
        self.BASE_URL = "https://autoai-backend-exjsxe2nda-uc.a.run.app/model"
        self.token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBOYW1lIjoiQUkgSGFuZCIsInVzZXJFbWFpbCI6ImFiaGlyYW0ua2FtaW5pQHRlY2hvbHV0aW9uLmNvbSIsInVzZXJJZCI6IjY1MWU1NTZjZWNhZGYzMjY5MzhlZWNkZCIsInNjb3BlT2ZUb2tlbiI6eyJwcm9qZWN0SWQiOiI2NGMxMGE2Mzk1MTEzMjc3OTI1YTgwZGYiLCJzY29wZXMiOnsicHJvamVjdCI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwicmVzb3VyY2UiOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX0sIm1vZGVsIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJkYXRhU2V0Q29sbGVjdGlvbiI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwibW9kZWxDb2xsZWN0aW9uIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJ0ZXN0Q29sbGVjdGlvbiI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwiY29waWxvdCI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwiY29waWxvdFJlc291cmNlIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJtb2RlbEdyb3VwIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9fX0sImlhdCI6MTczOTM3MzA1NX0.Q-CIyVoOAw6FmSIPeWn3_PGruLQLkp1e5xvfvSs4LHs"
        self.model_group_id = "678a262dc441e0b2c81a9686"
        self.project_id = "64c10a6395113277925a80df"
        self.task_type = "videoAnnotation"
        self.url = 'https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/'
        
    def fetch_tasks(self, task_name=None):
        """
        Fetch tasks based on model group ID and task name.

        Args:
            token (str): Authentication token
            model_group_id (str, optional): ID of the model group
            task_name (str, optional): Name of the task

        Returns:
            str: Task ID if found, None otherwise
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {}
        if self.model_group_id:
            params['modelGroupId'] = self.model_group_id
        if task_name:
            params['name'] = task_name

        try:
            response = requests.get(self.BASE_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data['models'][0]['_id'] if data.get('models') and data['models'] else None
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            return None

    def create_task(self,task_name):
        """
        Create a new task with specified parameters.

        Args:
            token (str): Authentication token
            task_name (str): Name of the task
            task_type (str): Type of the task
            model_group_id (str): ID of the model group
            project_id (str): ID of the project

        Returns:
            str: ID of created task, None if creation fails
        """

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "name": task_name,
            "type": self.task_type,
            "modelGroupId": self.model_group_id,
            "project": self.project_id
        }

        try:
            response = requests.post(self.BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json().get('_id')
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            return None

    def get_or_create_task(self,task_name):
        """
        Check if task exists, if not create new one.

        Args:
            token (str): Authentication token
            task_name (str): Name of the task
            task_type (str): Type of the task
            model_group_id (str): ID of the model group
            project_id (str): ID of the project

        Returns:
            str: Task ID (either existing or newly created)
        """
        existing_task_id = self.fetch_tasks(task_name)
        if existing_task_id:
            print(f"Task '{task_name}' already exists")
            return existing_task_id

        print(f"Creating new task '{task_name}'")
        return self.create_task(task_name)

    def convert_video(self, input_path, output_path):
        """
        Converts the video into the required format.
        """
        os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{output_path}'")

    def upload_to_rlef(self,filepath, task_id=None):
        """
        """
        self.filepath = filepath
        converted_filepath = f'{self.filepath}_converted.mp4'
        self.convert_video(self.filepath, converted_filepath)
        sample_folder = "/".join(self.filepath.split("/")[:-1])
        csv_file_path = os.path.join(sample_folder, 'augmented_predictions.csv')
        csv_file_name = f'{uuid.uuid4()}_hamer.csv'
        
        if task_id is None:
            print("Default Task ID Selected !")
            task_id = "67695dc462913593227a4227"
        annotations = {}
        payload = {
            'model': task_id,
            'status': 'backlog',
            'csv': "csv",
            'label': 'objects',
            'tag': 'boxes',
            'prediction': 'predicted',
            'confidence_score': '100',
            'videoAnnotations': annotations,
        }

        files = {
            'resource': (converted_filepath, open(converted_filepath, 'rb'))
        }

        response = requests.post(
            self.url, 
            headers={},
            data=payload,
            files=files
        )

        # Print the raw response for debugging
        print("-"*100)
        print(type(response.text))
        print(response.text)
        
        resource_id = response.json().get('_id')
        print(resource_id)
        
        signed_url = self.get_signed_url(resource_id,csv_file_name)
        if signed_url:
                print(f"Signed URL: {signed_url}")
                self.upload_hdf5_file(signed_url, csv_file_path)
        else:
            print("Error in Signed URL")
        if response.headers.get('Content-Type') == 'application/json':
            try:
                response_json = response.json()
                print("-"*100)
                print("Response:",response_json)  # Optionally print the parsed JSON response
            except ValueError as e:
                print(f"Failed to parse JSON: {e}")
        else:
            print(f"Unexpected Content-Type: {response.headers.get('Content-Type')}")

        return response.status_code

    def get_signed_url(self, resource_id, hdf5_filename):
        url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/uploadHdf5File"
        form_data = {
            "resourceId": resource_id,
            "hdf5FileName": hdf5_filename
        }
        response = requests.put(url, data=form_data)

        if response.status_code == 200:
            try:
                response_dict = response.json()
                return response_dict.get("hdf5FileSignedUrlForUpload")
            except json.JSONDecodeError:
                print("Error: Response is not valid JSON.")
                return None
        else:
            print(f"Failed to get signed URL. Status code: {response.status_code}")
            return None
    def upload_hdf5_file(self, signed_url, hdf5_filepath):
        headers = {"Content-Type": "text/csv"}

        with open(hdf5_filepath, 'rb') as file_data:
            response = requests.put(signed_url, headers=headers, data=file_data)

        print(f"Status Code: {response.status_code}")
        print(f"Response from the server: {response.text}")

    
        
if __name__ == "__main__":
    config = load_config("../config/config.yaml")['RLEF']
    manager = RLEFManager(config)
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBOYW1lIjoiQUkgSGFuZCIsInVzZXJFbWFpbCI6ImFiaGlyYW0ua2FtaW5pQHRlY2hvbHV0aW9uLmNvbSIsInVzZXJJZCI6IjY1MWU1NTZjZWNhZGYzMjY5MzhlZWNkZCIsInNjb3BlT2ZUb2tlbiI6eyJwcm9qZWN0SWQiOiI2NGMxMGE2Mzk1MTEzMjc3OTI1YTgwZGYiLCJzY29wZXMiOnsicHJvamVjdCI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwicmVzb3VyY2UiOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX0sIm1vZGVsIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJkYXRhU2V0Q29sbGVjdGlvbiI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwibW9kZWxDb2xsZWN0aW9uIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJ0ZXN0Q29sbGVjdGlvbiI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwiY29waWxvdCI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwiY29waWxvdFJlc291cmNlIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJtb2RlbEdyb3VwIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9fX0sImlhdCI6MTczOTM3MzA1NX0.Q-CIyVoOAw6FmSIPeWn3_PGruLQLkp1e5xvfvSs4LHs"
    task_name = "Pouring"
    task_type = "videoAnnotation"
    model_group_id = "67b5ac103336289948685dc4"
    project_id = "64c10a6395113277925a80df"
    task_id = model_group_id
 
    filepath = "../../Realtime-WebRTC/data/recordings/Aditi/sample_1/Aditi_video.mp4"
    manager.upload_to_rlef(filepath, task_id)