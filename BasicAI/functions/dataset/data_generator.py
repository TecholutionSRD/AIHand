"""
Author: Shreyas Dixit
This file is the main extraction file used to extract action trajectories from videos.
"""
import csv
import os
import sys
import concurrent.futures
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from BasicAI.functions.dataset.gemini_object_detector import ObjectDetector
from BasicAI.functions.dataset.utils import base64_to_csv_path, create_zip_archive, deproject_pixel_to_point, get_base64_encoded_hamer_response
from Config.config import load_config


class VideoProcessor4D:
    """
    A class for processing 4D video data with advanced coordinate transformation and object detection capabilities.
    This class handles video processing tasks including frame extraction, coordinate transformation,
    and object detection using various APIs and processing pipelines.
    Attributes:
        config (dict): Configuration settings loaded from the config file
        base_recordings_dir (str): Base directory for recordings
        classes (list): List of classes for object detection
        url (str): URL endpoint for processing
        fps (int): Frames per second for video processing
        num_samples (int): Number of samples to process
        final_csv (str): Path to the final CSV output
        gemini_api_key (str): API key for Gemini services
    Methods:
        configure_recording(classes, action): Configure recording settings for specific classes and action
        count_recording_samples(): Count the number of recording samples in the base directory
        process_recent_recording(action): Process the most recent recording for a given action
        process_all_videos(action): Process all videos for a specific action
        apply_coordinate_transformation(x, y, z): Apply coordinate transformation to 3D points
        extract_frames_and_generate_zip(recording_dir, fps): Extract frames and create zip archives
        detect_initial_object_centers(recording_dir, classes): Detect initial object centers using Gemini API
        run_complete_processing_pipeline(recording_dir, fps, classes, output_csv_path, action): Execute complete processing pipeline
    Raises:
        ValueError: If Gemini API key is not set or transformation matrices are not found
    """
    
    #-------------------------------------------------------------------------------------------#
    def __init__(self, config_path: str):
        """
        Initialize the DataGenerator with configuration parameters.
        The DataGenerator class is responsible for setting up data generation parameters
        from a configuration file for 4D hand gesture data.
        Args:
            config_path (str): Path to the configuration file containing data generation parameters.
        Raises:
            ValueError: If the Gemini API key environment variable is not set.
        Attributes:
            config (dict): Configuration dictionary for 4D data generation.
            base_recordings_dir (str): Base directory for storing recordings.
            classes (list): List of gesture classes to generate data for.
            url (str): URL for data access.
            fps (int): Frames per second for recordings.
            num_samples (int): Number of samples to generate.
            final_csv (str): Path to the final CSV output file.
            gemini_api_key (str): API key for Gemini service access.
        """
        self.config = load_config(config_path)['4D']
        self.base_recordings_dir = self.config["base_recordings_dir"]
        self.classes = self.config["classes"]
        self.url = self.config["url"]
        self.fps = self.config["fps"]
        self.num_samples = self.config['num_samples']
        self.final_csv = self.config['final_csv']
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.gemini_api_key:
            raise ValueError("[HaMeR] Gemini API is not set.")

    #-------------------------------------------------------------------------------------------#
    def configure_recording(self, classes: list[tuple[str, str]], action: str) -> None:
        """
        Configures the recording settings for the dataset.
        Args:
            classes (list[tuple[str, str]]): A list of tuples where each tuple contains
                                             the class name and its corresponding label.
            action (str): The action name to be used for setting the base recordings directory.
        Returns:
            None
        """
        self.classes = classes
        self.base_recordings_dir = os.path.join(self.config['base_recordings_dir'], action)
        print(f"[Hamer] Set base recordings directory: {self.base_recordings_dir}")

    #-------------------------------------------------------------------------------------------#
    def count_recording_samples(self) -> int:
        """
        Counts the number of recording samples in the base recordings directory.
        This method checks the base recordings directory for subdirectories,
        which represent individual recording samples. If the directory does not
        exist, it prints a message and returns 0.
        Returns:
            int: The number of recording samples (subdirectories) in the base recordings directory.
        """
        action_dir = self.base_recordings_dir
        if not os.path.exists(action_dir):
            print(f"[Hamer] Action directory {action_dir} does not exist.")
            return 0
        return sum(os.path.isdir(os.path.join(action_dir, name)) for name in os.listdir(action_dir))

    #-------------------------------------------------------------------------------------------#
    def process_recent_recording(self, action: str):
        """
        Processes the most recent recording directory.
        This method identifies the most recently modified directory within the base recordings directory,
        and runs a complete processing pipeline on the contents of that directory. The results are saved
        to a CSV file named 'processed_predictions_hamer.csv' within the latest directory.
        Args:
            action (str): The action to be performed during the processing pipeline.
        Returns:
            The result of the processing pipeline if successful, otherwise None.
        Raises:
            Exception: If an error occurs during the processing pipeline, it will be caught and printed.
        """
        subfolders = [
            os.path.join(self.base_recordings_dir, d)
            for d in os.listdir(self.base_recordings_dir)
            if os.path.isdir(os.path.join(self.base_recordings_dir, d))
        ]
        if not subfolders:
            print("[Hamer] No recording directories found.")
            return None

        latest_folder = max(subfolders, key=os.path.getmtime)
        output_csv_path = os.path.join(latest_folder, "processed_predictions_hamer.csv")

        try:
            return self.run_complete_processing_pipeline(latest_folder, self.fps, self.classes, output_csv_path, action)
        except Exception as e:
            print(f"[Hamer] Error processing {latest_folder}: {e}")
            return None

    #-------------------------------------------------------------------------------------------#
    def process_all_videos(self, action: str):
        """
        Processes all video files in the specified action directory.
        This method iterates through all subdirectories within the given action directory,
        processes each video file found, and generates a CSV file with the processed predictions.
        Args:
            action (str): The name of the action directory to process.
        Raises:
            Exception: If an error occurs during the processing of a video file, it is caught and logged.
        """
        filedir = Path(self.base_recordings_dir, action)
        for folder_name in os.listdir(filedir):
            subfolder_path = os.path.join(filedir, folder_name)
            if not os.path.isdir(subfolder_path):
                continue

            print(f"[Hamer] Processing {subfolder_path}")
            output_csv_path = os.path.join(subfolder_path, "processed_predictions_hamer.csv")

            try:
                self.run_complete_processing_pipeline(subfolder_path, self.fps, self.classes, output_csv_path, action)
            except Exception as e:
                print(f"[Hamer] Error processing {subfolder_path}: {e}")

    #-------------------------------------------------------------------------------------------#
    def apply_coordinate_transformation(self, x: float, y: float, z: float):
        """
        Applies a coordinate transformation to the given x, y, z coordinates using
        transformation matrices specified in the configuration.
        Args:
            x (float): The x-coordinate to be transformed.
            y (float): The y-coordinate to be transformed.
            z (float): The z-coordinate to be transformed.
        Returns:
            tuple: A tuple containing the transformed x, y, z coordinates.
        Raises:
            ValueError: If the transformation matrices for India are not found in the config.
        """
        transformation = self.config.get("Transformation", {}).get("India", {})
        if not transformation:
            raise ValueError("Transformation matrices for India not found in config.")
        X, Y = np.array(transformation["X"]), np.array(transformation["Y"])
        B = np.eye(4)
        B[:3, 3] = [x / 1000, y / 1000, z / 1000]
        A = Y @ B @ np.linalg.inv(X)
        return tuple(A[:3, 3] * 1000)

    #-------------------------------------------------------------------------------------------#
    def extract_frames_and_generate_zip(self, recording_dir: Path, fps: int = 5):
        """
        Extract frames from the given recording directory, generate zip archives for RGB and depth images,
        and process the zipped files to generate a CSV file with predictions.
        Args:
            recording_dir (Path): The directory containing the recording files.
            fps (int, optional): Frames per second for frame extraction. Defaults to 5.
        Returns:
            Path or None: The path to the generated CSV file with predictions if successful, otherwise None.
        """
        print("[Hamer] Extracting frames and processing")
        
        rgb_dir, depth_dir = Path(recording_dir, 'rgb'), Path(recording_dir, 'depth')
        rgb_zip_path, depth_zip_path = Path(recording_dir, 'rgb_images_collection.zip'), Path(recording_dir, 'depth_images_collection.zip')
        
        create_zip_archive(rgb_dir, rgb_zip_path)
        create_zip_archive(depth_dir, depth_zip_path)

        response_encoded = get_base64_encoded_hamer_response(str(rgb_zip_path), str(depth_zip_path), self.url)
        return base64_to_csv_path(response_encoded['csv_data'], f'{recording_dir}/predictions_hamer.csv') if response_encoded else None

    #-------------------------------------------------------------------------------------------#
    def detect_initial_object_centers(self, recording_dir: Path, classes: list[str]):
        """
        Detects the initial centers of specified objects in a recording directory.
        Args:
            recording_dir (Path): The directory containing the recording data.
            classes (list[str]): A list of object class names to detect.
        Returns:
            dict: A dictionary where keys are object class names and values are their 3D coordinates after applying coordinate transformation.
        """
        gemini_model = ObjectDetector(api_key=self.gemini_api_key, recording_dir=recording_dir)
        object_centers = {}
        intrinsics = rs.intrinsics()
        intrinsics.width, intrinsics.height = 640, 480
        intrinsics.ppx, intrinsics.ppy = 329.13, 240.29
        intrinsics.fx, intrinsics.fy = 611.08, 609.76
        intrinsics.model, intrinsics.coeffs = rs.distortion.inverse_brown_conrady, [0, 0, 0, 0, 0]
        
        for name in classes:
            center_x, center_y, *_ = gemini_model.get_object_center(name)
            point_3d = deproject_pixel_to_point((center_x, center_y), intrinsics)
            object_centers[name] = self.apply_coordinate_transformation(*point_3d)
        
        return object_centers

    #-------------------------------------------------------------------------------------------#
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
    #-------------------------------------------------------------------------------------------#
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
                file_path = os.path.join(base_recordings_dir, folder_name,"processed_predictions_hamer.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    merged_data = pd.concat([merged_data, df], ignore_index=True)

        output_file_path = os.path.join(base_recordings_dir, self.final_csv)
        merged_data.to_csv(output_file_path, index=False)
        print(f"[Hamer] Merged CSV saved to {output_file_path}")

    #-------------------------------------------------------------------------------------------#
    def run_complete_processing_pipeline(self, recording_dir: Path, fps: int, classes: list[str], output_csv_path: Path, action: str) -> Path:
        """
        Executes the complete processing pipeline for a given recording directory.
        This method performs the following steps:
        1. Extracts frames from the recording and generates a zip file.
        2. Detects initial object centers in the recording based on the provided classes.
        3. Finalizes the CSV output with the detected object centers.
        4. Merges all processed CSV files based on the specified action.
        Args:
            recording_dir (Path): The directory containing the recording files.
            fps (int): Frames per second to extract from the recording.
            classes (list[str]): List of class names to detect in the recording.
            output_csv_path (Path): The path where the final CSV output will be saved.
            action (str): The action to be performed during the merging of processed CSV files.
        Returns:
            Path: The path to the final CSV output file.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_csv_path = executor.submit(self.extract_frames_and_generate_zip, recording_dir, fps)
            future_input_dict = executor.submit(self.detect_initial_object_centers, recording_dir, classes)

            csv_path, input_dict = future_csv_path.result(), future_input_dict.result()
        
        self.finalize_csv_output(csv_path, output_csv_path, input_dict)
        self.merge_all_processed_csv(base_recordings_dir=self.base_recordings_dir, action_name=action)
        
        return output_csv_path  