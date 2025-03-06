"""
Author: Shreyas Dixit
This file contains the PreProcessor class to filter, rename, and merge CSV data.
"""
import os
import shutil
import json
from pathlib import Path
from BasicAI.functions.dataset.data_generator import VideoProcessor4D
from BasicAI.functions.dataset.augmentation import DataAugmenter
from Config.config import load_config

class PreProcessor:
    """Class to preprocess action data by filtering, renaming, and merging CSV files."""

    def __init__(self, config_path: str):
        """
        Initialize the PreProcessor with configuration.

        Args:
            config_path (str): Path to the configuration JSON file.
        """
        self.config = load_config(config_path)['4D']
        self.base_dir = Path(self.config.get('base_recordings_dir'))
        self.processor = VideoProcessor4D(config_path)

    def process(self, action_name: str) ->  bool:
        """
        Main function to preprocess data and generate merged CSV.
        
        Args:
            action_name (str): The name of the action being processed.
        """
        print(f"[PreProcessor] Processing action: {action_name}")
        self.action_dir = Path(f"{self.base_dir}/{action_name}")
        if os.path.exists(self.action_dir):
            self.merged_csv_path = Path(f"{self.base_dir}/{action_name}/merged_predictions_hamer.csv")
            self.action_dir = Path(f"{self.base_dir}/{action_name}")
            if self.merged_csv_path.exists():
                print("[PreProcessor] Merged CSV already exists. Skipping processing.")
                return True

            self._filter_valid_folders()
            self._rename_folders()
            self._generate_merged_csv()
        else:
            print("[PreProcessor] Action is not available !")
            return False

    def _filter_valid_folders(self):
        """Remove subfolders that do not contain the required `processed_predictions_hamer.csv`."""
        print("[PreProcessor] Filtering subfolders...")

        for folder in self.action_dir.iterdir():
            if folder.is_dir():
                csv_file = folder / "processed_predictions_hamer.csv"
                if not csv_file.exists():
                    print(f"[PreProcessor] Removing {folder} (missing CSV)")
                    shutil.rmtree(folder)

        print("[PreProcessor] Filtering complete.")

    def _rename_folders(self):
        """Rename subfolders sequentially as `sample_1`, `sample_2`, ... `sample_n`."""
        print("[PreProcessor] Renaming subfolders...")

        subfolders = sorted([f for f in self.action_dir.iterdir() if f.is_dir()])
        for index, folder in enumerate(subfolders, start=1):
            new_name = self.action_dir / f"sample_{index}"
            if folder != new_name:
                print(f"[PreProcessor] Renaming {folder} → {new_name}")
                folder.rename(new_name)

        print("[PreProcessor] Renaming complete.")

    def _generate_merged_csv(self):
        """Use `VideoProcessor4D` to generate the merged CSV file."""
        print("[PreProcessor] Generating merged CSV...")
        #TODO: Pass main folder not action.
        self.processor.merge_all_processed_csv(self.action_dir,action)
        print(f"[PreProcessor] Merged CSV saved at {self.merged_csv_path}")

    async def process_hamer(self, action_name: str, objects: list[tuple[str, str]]):
        """
        Asynchronously processes a recording action and augments the data.
        Args:
            action_name (str): The name of the action to be processed.
            objects (list[str]): A list of objects related to the action.
        Returns:
            dict: A dictionary containing a message about the processing status and the number of samples processed.
        """
        try:
            self.processor.configure_recording(objects, action_name)
            action_dir = f"data/recordings/{action_name}/"
            
            num_samples = sum(1 for d in os.listdir(action_dir) if d.startswith("sample_") and os.path.isdir(os.path.join(action_dir, d)))
            
            await self.processor.process_recent_recording(action_name)
            
            try:
                csv_path = f"{action_dir}/sample_{num_samples}/processed_predictions_hamer.csv"
                output_path = f"{action_dir}/sample_{num_samples}/boosted_predictions.csv"

                augmenter = DataAugmenter(input_csv=csv_path, output_csv=output_path)
                augmenter.augment_data()
                augmenter.save_to_csv()
                
                print(f"[PreProcessor] Data no {num_samples} collected for {action_name} !")
                return {"status": "success","message": "[PreProcessor] Processing and data boost completed successfully","num_samples": num_samples}

            except Exception as e:
                print(f"[PreProcessor] Data collected but Data Boost failed: {str(e)}")
                return {"status": "partial_success","message": "[PreProcessor] Hammer processing succeeded, but data boost failed","error": str(e),"num_samples": num_samples}
            
        except Exception as e:
            print(f"[PreProcessor] Processing failed: {str(e)}")
            return {"status": "error","message": "[PreProcessor] Processing failed","error": str(e)}
