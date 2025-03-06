"""
Author: Shreyas Dixit
This file is the main extraction file used to extract action trajectories from videos.
"""
import os
import json
import pprint
import tempfile
import shutil
import glob
import concurrent
import h5py
import sys
import numpy as np
from PIL import Image
import pandas as pd
import pyrealsense2 as rs
import csv
from pathlib import Path
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from BasicAI.functions.dataset.gemini_object_detector import ObjectDetector
from BasicAI.functions.dataset.utils import base64_to_csv_path, create_zip_archive,deproject_pixel_to_point,get_base64_encoded_hamer_response
from Config.config import load_config

class VideoProcessor4D:
    def __init__(self, config_path:str):
        """
        Initializes the data generator with the given configuration.
        Args:
            config_path (Path): Path to the configuration file.
        Attributes:
            config (str): Loaded configuration dictionary.
            base_recordings_dir (str): Directory path for base recordings.
            classes (list): List of classes.
            url (str): URL for data source.
            fps (int): Frames per second for recordings.
            gemini_api_key (str): API key for Gemini service.
            num_samples (int): Number of samples to generate.
            final_csv (str): Path to the final CSV file.
        """
        self.config = load_config(config_path)['4D']
        self.base_recordings_dir = self.config["base_recordings_dir"]
        self.classes = self.config["classes"]
        self.url = self.config["url"]
        self.fps = self.config["fps"]
        self.num_samples = self.config['num_samples']
        self.final_csv = self.config['final_csv']
        if os.getenv('GEMINI_API_KEY') is None:
            raise ValueError("[HaMeR] Gemini API is not set.")
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')


    def configure_recording(self,classes:list[tuple[str,str]], action:str) -> None:
        """
        Sets the details for the data generator.
        Args:
            classes (list[tuple[str, str]]): A list of tuples where each tuple contains two strings representing class details.
            action (str): The action name to be used for setting the base recordings directory.
        Sets:
            self.classes: The provided list of class details.
            self.base_recordings_dir: The directory path for base recordings, constructed using the base path from the configuration and the provided action name.
        Prints:
            The constructed base recordings directory path.
        """
        self.classes = classes
        self.base_recordings_dir = f"{self.config['base_recordings_dir']}/{action}"
        print(f"[Hamer] Set base recordings directory: {self.base_recordings_dir}")
    
    def count_recording_samples(self) -> int:
        """
        Checks the number of available samples in the action directory.
        This method counts the number of subdirectories within the base recordings directory,
        which represent the available samples.
        Returns:
            int: The number of available samples. Returns 0 if the action directory does not exist.
        """
        action_dir = self.base_recordings_dir
        if not os.path.exists(action_dir):
            print(f"[Hamer] Action directory {action_dir} does not exist.")
            return 0
        return len([name for name in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, name))])

    def process_recent_recording(self, action: str):
        """
        Processes the latest recording by selecting the most recently modified subfolder
        in the base recordings directory, and then running a combined pipeline on the 
        recording data.
        Args:
            action (str): The action to be performed by the combined pipeline.
        Returns:
            str or None: The path to the processed CSV file if successful, or None if an error occurs
            or no recording directories are found.
        Raises:
            Exception: If an error occurs during the execution of the combined pipeline.
        """
        subfolders = [os.path.join(self.base_recordings_dir, d) for d in os.listdir(self.base_recordings_dir) if os.path.isdir(os.path.join(self.base_recordings_dir, d))]
        if not subfolders:
            print("[Hamer] No recording directories found.")
            return None

        latest_folder = max(subfolders, key=os.path.getmtime)
        output_csv_path = os.path.join(latest_folder, "processed_predictions_hamer.csv")

        try:
            return self.run_complete_processing_pipeline(latest_folder, self.fps, self.classes, output_csv_path, action)
        except Exception as e:
            print(f"[Hamer] Error running run_complete_processing_pipeline for {latest_folder}:\n{e}")
            return None

    def process_all_videos(self,action:str):
        """
        Processes all recordings in the base recordings directory by iterating through each subfolder.
        Args:
            action (str): The action to be performed during the processing of recordings.
        This method performs the following steps:
        1. Iterates through each folder in the base recordings directory.
        2. Checks if the current item is a directory.
        3. Prints the path of the directory being processed.
        4. Constructs the output CSV file path for processed predictions.
        5. Calls the run_complete_processing_pipeline method with the subfolder path, frames per second (fps), classes, output CSV path, and action.
        6. Catches and prints any exceptions that occur during the execution of the run_complete_processing_pipeline method.
        Note:
            The run_complete_processing_pipeline method is expected to be defined elsewhere in the class.
        """
        filedir = Path(f"{self.base_recordings_dir}/{action}")
        for folder_name in os.listdir(filedir):
            subfolder_path = os.path.join(filedir, folder_name)
            if not os.path.isdir(subfolder_path):
                continue

            print(f"[Hamer] Processing {subfolder_path}")
            output_csv_path = os.path.join(subfolder_path, "processed_predictions_hamer.csv")
        
            try:
                self.run_complete_processing_pipeline(subfolder_path, self.fps, self.classes, output_csv_path, action)
            except Exception as e:
                print(f"[Hamer] Error running run_complete_processing_pipeline for {subfolder_path}:\n{e}")


    def apply_coordinate_transformation(self, x: float, y: float, z: float):
        """
        Transforms the given coordinates (x, y, z) using transformation matrices specified in the configuration.
        Args:
            x (float): The x-coordinate to be transformed.
            y (float): The y-coordinate to be transformed.
            z (float): The z-coordinate to be transformed.
        Returns:
            tuple: A tuple containing the transformed coordinates (transformed_x, transformed_y, transformed_z).
        Raises:
            ValueError: If the transformation matrices for India are not found in the configuration.
        """
        transformation = self.config.get("Transformation", {}).get("India", {})
        if not transformation:
            raise ValueError("Transformation matrices for India not found in config.")
        X = np.array(transformation["X"])
        Y = np.array(transformation["Y"])
        B = np.eye(4)
        B[:3, 3] = [x / 1000, y / 1000, z / 1000]
        A = Y @ B @ np.linalg.inv(X)
        transformed_x, transformed_y, transformed_z = A[:3, 3] * 1000

        return transformed_x, transformed_y, transformed_z


    def extract_frames_and_generate_zip(self, recording_dir: Path, fps: int = 5):
        """
        Extracts frames from RGB and depth recordings, processes them, and creates ZIP archives.
        This method performs the following steps:
        1. Extracts frames from the given recording directory.
        2. Copies the frames to temporary directories.
        3. Creates ZIP archives of the extracted frames.
        4. Sends the ZIP archives to a remote server for processing.
        5. Decodes the server response and saves the predictions to a CSV file.
        Args:
            recording_dir (Path): The directory containing the RGB and depth recordings.
            fps (int, optional): Frames per second for processing. Defaults to 5.
        Returns:
            Optional[Path]: The path to the CSV file containing the predictions, or None if processing failed.
        """
        print("[Hamer] Extracting frames and processing")

        temp_rgb_dir, temp_depth_dir = os.path.join(tempfile.gettempdir(), 'temp_rgb'), os.path.join(tempfile.gettempdir(), 'temp_depth')
        os.makedirs(temp_rgb_dir, exist_ok=True)
        os.makedirs(temp_depth_dir, exist_ok=True)
        temp_rgb_dir
        rgb_dir, depth_dir = os.path.join(recording_dir, 'rgb'), os.path.join(recording_dir, 'depth')
        rgb_files, depth_files = glob.glob(os.path.join(rgb_dir, '*.[pj][np][g]*')), glob.glob(os.path.join(depth_dir, '*.npy'))

        for i, rgb_path in enumerate(sorted(rgb_files)):
            shutil.copy(rgb_path, os.path.join(temp_rgb_dir, f'image_{i}.png'))
        for i, depth_path in enumerate(sorted(depth_files)):
            shutil.copy(depth_path, os.path.join(temp_depth_dir, f'image_{i}.npy'))

        temp_rgb_dir = f"{recording_dir}/rgb/"
        temp_depth_dir = f"{recording_dir}/depth"
        rgb_zip_path, depth_zip_path = os.path.join(recording_dir, 'rgb_images_collection.zip'), os.path.join(recording_dir, 'depth_images_collection.zip')
        create_zip_archive(temp_rgb_dir, rgb_zip_path)
        create_zip_archive(temp_depth_dir, depth_zip_path)

        print(f"[Hamer] Created ZIPs: {rgb_zip_path}, {depth_zip_path}")

        response_encoded = get_base64_encoded_hamer_response(rgb_zip_path, depth_zip_path, self.url)

        if response_encoded:
            return base64_to_csv_path(response_encoded['csv_data'], f'{recording_dir}/predictions_hamer.csv')

        return None

    def detect_initial_object_centers(self, recording_dir: Path, classes: list[str]):
        """
        Initializes the centers of objects in the initial frame of a recording and plots detected centers.
        Args:
            recording_dir (Path): The directory containing the recording data.
            classes (list[str]): A list of object class names to detect.
        Returns:
            dict: A dictionary where keys are object class names and values are lists of transformed 3D coordinates [x, y, z].
        """
        gemini_model = ObjectDetector(api_key=os.getenv('GEMINI_API_KEY'), recording_dir=recording_dir)
        object_centers = {}
        rgb_image_path = next((f"{recording_dir}/initial_frames/image_0{ext}" for ext in ['.jpg', '.jpeg', '.png'] if os.path.exists(f"{recording_dir}/initial_frames/image_0{ext}")), None)
        depth_image_path = f"{recording_dir}/initial_frames/image_0.npy"

        if not rgb_image_path or not os.path.exists(depth_image_path):
            print("[Hamer] Warning: Missing RGB or depth image for detection.")
            return {}

        try:
            rgb_image = Image.open(rgb_image_path)
            depth_image = np.load(depth_image_path)
            # Convert PIL Image to OpenCV format for drawing
            cv_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[Hamer] Error loading images: {e}")
            return {}

        intrinsics = rs.intrinsics()
        intrinsics.width, intrinsics.height, intrinsics.ppx, intrinsics.ppy = 640, 480, 329.1317443847656, 240.29669189453125
        intrinsics.fx, intrinsics.fy = 611.084594726562, 609.7639770507812
        intrinsics.model, intrinsics.coeffs = rs.distortion.inverse_brown_conrady, [0, 0, 0, 0, 0]

        # Draw centers on image
        for name in classes:
            center_x, center_y, _, _ = gemini_model.get_object_center(rgb_image, name)
            print(f"[Hamer] {name} Object Center : {center_x}, {center_y}")
            point_3d = deproject_pixel_to_point(depth_image, (center_x, center_y), intrinsics)
            print(f"[Hamer] {name} 3D Object Center : {point_3d[0]}, {point_3d[1]}, {point_3d[2]}")
            transformed_x, transformed_y, transformed_z = self.apply_coordinate_transformation(*point_3d)
            print(f"[Hamer] {name} Transformed :", transformed_x, transformed_y, transformed_z)
            object_centers[name] = [transformed_x, transformed_y, transformed_z]
            
            # Draw center point and label
            cv2.circle(cv_image, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
            cv2.putText(cv_image, name, (int(center_x)+10, int(center_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save annotated image
        output_path = os.path.join(recording_dir, "initial_frames", "detected_centers.jpg")
        cv2.imwrite(output_path, cv_image)
        print(f"[Hamer] Saved annotated image to {output_path}")

        return object_centers
    
    def finalize_csv_output(self, input_csv_path: Path, output_csv_path: Path, input_dict: dict):
        """
        Post-processes a CSV file by reading data from an input CSV file, modifying it based on the provided input dictionary,
        and writing the processed data to an output CSV file.
        Args:
            input_csv_path (Path): Path to the input CSV file.
            output_csv_path (Path): Path to the output CSV file.
            input_dict (dict): Dictionary containing initial input values for processing.
        The function performs the following steps:
        1. Reads the input CSV file.
        2. Initializes the fieldnames for the output CSV file.
        3. Writes the header to the output CSV file.
        4. Initializes row data with default values.
        5. Updates row data with values from the input dictionary.
        6. Iterates through the rows of the input CSV file and updates row data with values from each row.
        7. Writes the processed row data to the output CSV file.
        """
        with open(input_csv_path, 'r') as infile, open(output_csv_path, 'w', newline='') as outfile:
            reader, fieldnames = csv.DictReader(infile), ['input0_x', 'input0_y', 'input0_z', 'input1_x', 'input1_y', 'input1_z']
            for i in range(self.num_samples):
                fieldnames.extend([f'p{i}_x', f'p{i}_y', f'p{i}_z', f'p{i}_c'])

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            row_data = {field: 0 for field in fieldnames}

            object_keys = list(input_dict.keys())
            row_data['input0_x'], row_data['input0_y'], row_data['input0_z'] = input_dict.get(object_keys[1], [0, 0, 0])
            row_data['input1_x'], row_data['input1_y'], row_data['input1_z'] = input_dict.get(object_keys[0], [0, 0, 0])

            for i, row in enumerate(reader):
                row_data[f'p{i}_x'], row_data[f'p{i}_y'], row_data[f'p{i}_z'], row_data[f'p{i}_c'] = row.get('X', 0), row.get('Y', 0), row.get('Z', 0), row.get('C', 0)

            writer.writerow(row_data)

    def merge_all_processed_csv(self, base_recordings_dir: Path, action_name:str):
        """
        Combines multiple CSV files from subdirectories within the base_recordings_dir into a single CSV file.
        This method searches for CSV files named "processed_predictions_hamer.csv" in each subdirectory of the 
        specified base_recordings_dir. It reads these CSV files, concatenates their contents into a single DataFrame, 
        and then saves the merged DataFrame to a new CSV file specified by self.final_csv.
        Args:
            base_recordings_dir (Path): The base directory containing subdirectories with CSV files to be combined.
        Returns:
            None
        """
        print("[Hamer] Combining CSV files")

        merged_data = pd.DataFrame()
        for folder_name in os.listdir(base_recordings_dir):
                file_path = os.path.join(base_recordings_dir, folder_name, action_name,"processed_predictions_hamer.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    merged_data = pd.concat([merged_data, df], ignore_index=True)

        output_file_path = os.path.join(base_recordings_dir, self.final_csv)
        merged_data.to_csv(output_file_path, index=False)
        print(f"[Hamer] Merged CSV saved to {output_file_path}")


    def run_complete_processing_pipeline(self, recording_dir: Path, fps: int, classes: list[str], output_csv_path: Path, action: str) -> Path:
        """
        Combined pipeline for video analysis, object detection, and coordinate transformation.
        
        Args:
            recording_dir (Path): Path to recording directory.
            fps (int): Frames per second.
            classes (list[str]): List of object classes to detect.
            output_csv_path (Path): Path to output CSV.
            action (str): Action parameter for CSV processing.

        Returns:
            Path: Path to the processed CSV file.
        """
        print("\n[Hamer] Starting extractions and detect_initial_object_centers processes in parallel...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_csv_path = executor.submit(self.extract_frames_and_generate_zip, recording_dir, fps)
            future_input_dict = executor.submit(self.detect_initial_object_centers, recording_dir, classes)

            try:
                csv_path = future_csv_path.result()
                input_dict = future_input_dict.result()
            except Exception as e:
                print(f"[Hamer] Error in parallel execution: {e}")
                return None

        print(f"[Hamer] extracttion completed successfully: {csv_path}")
        print(f"[Hamer] detect_initial_object_centers completed successfully: {input_dict}\n")

        print("[Hamer] Starting finalize_csv_output...")
        try:
            self.finalize_csv_output(csv_path, output_csv_path, input_dict)
            print(f"[Hamer] finalize_csv_output completed successfully: {output_csv_path}\n")
        except Exception as e:
            print(f"[Hamer] Error in finalize_csv_output: {e}")
            return None

        print("[Hamer] Starting merge_all_processed_csv...")
        try:
            self.merge_all_processed_csv(self.base_recordings_dir, action)
            print(f"[Hamer] merge_all_processed_csv completed successfully \n")
        except Exception as e:
            print(f"[Hamer] Error in merge_all_processed_csv: {e}")
            return None

        print("-" * 100)
        print("[Hamer] All processing completed successfully.")

        return output_csv_path


if __name__ == "__main__":
    config_path = "../config/config.yaml"
    processor = VideoProcessor4D(config_path)
    processor.process_all_videos()
