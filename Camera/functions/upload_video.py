"""
This file contains the code to process and upload task videos to Google Cloud Storage.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from dotenv import load_dotenv
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config

config = load_config("camera_config.yaml")['GCP_BUCKET_JSON']
BUCKET_NAME = config.get("bucket_name", "video-analysing")

load_dotenv()

GCP_CREDENTIALS_PATH = os.getenv("GCP_CREDENTIALS_PATH")

if not GCP_CREDENTIALS_PATH:
    raise EnvironmentError("GCP_CREDENTIALS_PATH is not set in the .env file.")

try:
    with open(GCP_CREDENTIALS_PATH, "r") as f:
        GCP_CREDENTIALS = json.load(f)
    print("GCP credentials loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"GCP credentials file not found at {GCP_CREDENTIALS_PATH}")
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON format in {GCP_CREDENTIALS_PATH}")


def upload_video(source_path: str) -> Optional[str]:
    """
    Upload a video file to Google Cloud Storage after conversion.
    Args:
        source_path (str): Local path to the video file.
    Returns:
        Optional[str]: Public URL of the uploaded video if successful, None otherwise.
    """
    try:
        source_path = Path(source_path)
        if not source_path.exists():
            print(f"[Upload GCP] Source file not found: {source_path}")
            return None

        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        if source_path.suffix.lower() not in valid_extensions:
            print(f"[Upload GCP] Invalid video format {source_path.suffix}. Supported formats: {valid_extensions}")
            return None

        converted_path = convert_video(str(source_path))
        if not converted_path:
            print("[Upload GCP] Video conversion failed.")
            return None

        converted_source_path = Path(converted_path)
        parts = converted_source_path.parts
        try:
            recordings_idx = parts.index('recordings')
            action, sample, video_name = parts[recordings_idx + 1:recordings_idx + 4]
            destination_blob_name = f"recordings/{action}/{sample}/{video_name}"
        except ValueError:
            print("[Upload GCP] Invalid file path structure for GCS storage.")
            return None

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        blob = bucket.blob(destination_blob_name)
        blob.content_type = 'video/mp4'
        blob.upload_from_filename(converted_path)

        public_url = f"https://storage.googleapis.com/{bucket.name}/{destination_blob_name}"
        print(f"[Upload GCP] Video uploaded to {public_url}")

        return public_url
    except GoogleCloudError as gce:
        print(f"[Upload GCP] GCS Upload Error: {gce}")
        return None
    except Exception as e:
        print(f"[Upload GCP] Unexpected error: {e}")
        return None


def convert_video(input_path: str) -> Optional[str]:
    """
    Converts the video into the required format using ffmpeg.
    Args:
        input_path (str): Path to the input video file.
    Returns:
        Optional[str]: Path to the converted video if successful, None otherwise.
    """
    try:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.with_stem(f"{input_path_obj.stem}_converted").with_suffix(".mp4")
        if output_path.exists():
            print(f"[Upload GCP] Converted file already exists: {output_path}")
            return str(output_path)
        command = f"ffmpeg -i '{input_path}' -vcodec libx264 -acodec aac '{output_path}' -y"
        result = os.system(command)
        if result == 0:
            print(f"[Upload GCP] Video conversion completed: {output_path}")
            return str(output_path)
        else:
            print(f"[Upload GCP] Video conversion failed with exit code: {result}")
            return None
    except Exception as e:
        print(f"[Upload GCP] Video conversion error: {e}")
        return None