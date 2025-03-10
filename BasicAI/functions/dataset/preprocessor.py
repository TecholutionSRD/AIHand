import os
import shutil
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from BasicAI.functions.dataset.data_generator import VideoProcessor4D
from BasicAI.functions.dataset.augmentation import DataAugmenter
from Config.config import load_config

class PreProcessor():
    """
    A class for preprocessing action data, including recording, cleaning, and merging functionality.
    This class handles two main modes of operation:
    1. Standard Mode: Cleans existing data and merges CSV files
    2. Hamer Mode: Records new data, processes it through GCP, and performs data augmentation
    Attributes:
        config (dict): Configuration settings for 4D processing
        base_dir (Path): Base directory path for recordings
        processor (VideoProcessor4D): Instance of VideoProcessor4D for video processing
    Methods:
        process_action_data(action_name: str, objects: list[tuple[str, str]] = None, hamer: bool = False) -> dict:
            Process action data in either standard or Hamer mode.
        _filter_invalid_samples(action_dir: Path):
            Remove sample folders that don't contain required CSV files.
        _rename_samples(action_dir: Path):
            Rename sample folders to maintain sequential ordering.
        _generate_merged_csv(action_dir: Path, merged_csv_path: Path):
            Generate a merged CSV file from all processed samples.
    Example:
        >>> preprocessor = PreProcessor('config.json')
        >>> result = await preprocessor.process_action_data('wave_hand', hamer=False)
    """
    def __init__(self, config_path: str):
        """
        Initialize the PreProcessor with configuration.

        Args:
            config_path (str): Path to the configuration JSON file.
        """
        config = load_config(config_path)
        self.config = config.get('4D', {})
        self.base_dir = Path(self.config.get('base_recordings_dir'))
        self.processor = VideoProcessor4D(config_path)

    async def process_action_data(self, action_name: str, objects: list[tuple[str, str]] = None, hamer: bool = False) -> dict:
        """
        Processes action data for a given action name. Supports two modes: Hamer mode and Standard mode.
        Args:
            action_name (str): The name of the action to process.
            objects (list[tuple[str, str]], optional): A list of objects required for Hamer processing. Defaults to None.
            hamer (bool, optional): Flag to indicate if Hamer mode should be used. Defaults to False.
        Returns:
            dict: A dictionary containing the status and message of the processing result. 
                  In case of success, additional information such as the number of samples or the path to the merged CSV may be included.
        """
        action_dir = self.base_dir / action_name
        if not action_dir.exists():
            print(f"[PreProcessor] Action {action_name} does not exist!")
            return {"status": "error", "message": "Action directory not found."}

        #------------------ Hamer Mode: Recording, GCP Upload, Processing, Augmentation ------------------#
        if hamer:
            if not objects:
                return {"status": "error", "message": "Objects list is required for Hamer processing."}

            print(f"[PreProcessor] Starting Hamer processing for {action_name}...")
            try:
                self.processor.configure_recording(objects, action_name)
                
                num_samples = sum(1 for d in os.listdir(action_dir) if d.startswith("sample_") and os.path.isdir(action_dir / d))

                await self.processor.process_recent_recording(action_name)
                csv_path = action_dir / f"sample_{num_samples}/processed_predictions_hamer.csv"
                output_path = action_dir / f"sample_{num_samples}/boosted_predictions.csv"

                augmenter = DataAugmenter(input_csv=csv_path, output_csv=output_path)
                augmenter.augment_data()
                augmenter.save_to_csv()

                print(f"[PreProcessor] Data no {num_samples} collected and boosted for {action_name}!")
                return {
                    "status": "success",
                    "message": "Hamer processing and data augmentation completed successfully.",
                    "num_samples": num_samples
                }

            except Exception as e:
                print(f"[PreProcessor] Hamer processing failed: {str(e)}")
                return {
                    "status": "error",
                    "message": "Hamer processing failed.",
                    "error": str(e)
                }

         #------------------ Standard Mode: Cleaning, merging csv ------------------#
        print(f"[PreProcessor] Cleaning and merging CSVs for {action_name}...")

        merged_csv_path = action_dir / "merged_predictions_hamer.csv"
        if merged_csv_path.exists():
            print("[PreProcessor] Merged CSV already exists. Skipping processing.")
            return {"status": "success", "message": "Merged CSV already exists."}

        self._filter_invalid_samples(action_dir)
        self._rename_samples(action_dir)
        self._generate_merged_csv(action_dir, merged_csv_path, action_name)

        print(f"[PreProcessor] Processing complete. Merged CSV saved at {merged_csv_path}")
        return {
            "status": "success",
            "message": "Data cleaning and merging completed successfully.",
            "merged_csv_path": str(merged_csv_path)
        }

    def _filter_invalid_samples(self, action_dir: Path):
        """
        Remove folders that do not contain the required `processed_predictions_hamer.csv`.
        Args:
            action_dir (Path): The directory containing folders to be checked and filtered.
        """
        print("[PreProcessor] Filtering invalid samples...")

        for folder in action_dir.iterdir():
            if folder.is_dir():
                csv_file = folder / "processed_predictions_hamer.csv"
                if not csv_file.exists():
                    print(f"[PreProcessor] Removing {folder} (missing CSV)")
                    shutil.rmtree(folder)

        print("[PreProcessor] Filtering complete.")

    def _rename_samples(self, action_dir: Path):
        """
        Rename sample folders sequentially within the given action directory.
        This method renames all subdirectories within the specified `action_dir` 
        to follow a sequential naming pattern (e.g., sample_1, sample_2, ...).
        Args:
            action_dir (Path): The directory containing the sample folders to be renamed.
        """
        print("[PreProcessor] Renaming sample folders...")

        subfolders = sorted([f for f in action_dir.iterdir() if f.is_dir()])
        for index, folder in enumerate(subfolders, start=1):
            new_name = action_dir / f"sample_{index}"
            if folder != new_name:
                print(f"[PreProcessor] Renaming {folder} → {new_name}")
                folder.rename(new_name)

        print("[PreProcessor] Renaming complete.")

    def _generate_merged_csv(self, action_dir: Path, merged_csv_path: Path, action_name: str):
        """
        Generate a merged CSV file using `VideoProcessor4D`.

        This method processes all CSV files in the specified action directory
        and merges them into a single CSV file.

        Args:
            action_dir (Path): The directory containing the action CSV files to be merged.
            merged_csv_path (Path): The path where the merged CSV file will be saved.

        Returns:
            None
        """
        print("[PreProcessor] Generating merged CSV...")
        self.processor.merge_all_processed_csv(base_recordings_dir=action_dir, action_name=action_name)
        print(f"[PreProcessor] Merged CSV saved at {merged_csv_path}")

