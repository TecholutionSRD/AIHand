import cv2
import os
from pathlib import Path
import pandas as pd

# import os
# import zipfile
# from pathlib import Path

# def zip_folders(base_path):
#     # Convert string path to Path object
#     base_dir = Path(base_path)
    
#     # Check if base directory exists
#     if not base_dir.exists():
#         print(f"Error: Directory {base_path} does not exist")
#         return
    
#     # Iterate through all subfolders
#     for subfolder in base_dir.iterdir():
#         if subfolder.is_dir():
#             # Check for rgb and depth folders
#             rgb_folder = subfolder / 'rgb'
#             depth_folder = subfolder / 'depth'
            
#             # Create zip for rgb folder
#             if rgb_folder.exists():
#                 zip_path = subfolder / 'rgb.zip'
#                 with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#                     for file in rgb_folder.rglob('*'):
#                         if file.is_file():
#                             zipf.write(file, file.relative_to(rgb_folder))
#                 print(f"Created {zip_path}")
            
#             # Create zip for depth folder
#             if depth_folder.exists():
#                 zip_path = subfolder / 'depth.zip'
#                 with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#                     for file in depth_folder.rglob('*'):
#                         if file.is_file():
#                             zipf.write(file, file.relative_to(depth_folder))
#                 print(f"Created {zip_path}")

# if __name__ == "__main__":
#     recording_path = "data/recordings/sprinkle"
#     zip_folders(recording_path)

# # #-------------------------------------------------------------------------------------------#
# def get_frame_number(path):
#     return int(path.stem.split('_')[1])
    
# def create_video_from_images(folder_path, output_path, fps=2):
#     # Get list of image files with numerical sorting
#     images = sorted([img for img in Path(folder_path).glob('*.png')], key=get_frame_number)
    
#     print("-"*100)
#     for i in range(16):
#         print(f"Path : {str(images[i])}")
#     if not images:
#         print(f"No images found in {folder_path}")
#         return

#     # Read first image to get dimensions
#     frame = cv2.imread(str(images[0]))
#     height, width, _ = frame.shape

#     # Create video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Write frames to video
#     for img_path in images:
#         frame = cv2.imread(str(img_path))
#         out.write(frame)

#     out.release()
#     print(f"Created video: {output_path}")

# def create_objects_file(folder_path):
#     objects_path = folder_path / 'objects.txt'
#     with open(objects_path, 'w') as f:
#         f.write("red bottle\nslices of bread")
#     print(f"Created objects file: {objects_path}")

# def process_all_folders(base_path="data/recordings/sid"):
#     base_dir = Path(base_path)
#     for folder in base_dir.iterdir():
#         if folder.is_dir():
#             rgb_path = folder / 'rgb'
#             if rgb_path.exists():
#                 video_path = folder / 'sid_video.mp4'
#                 create_video_from_images(rgb_path, str(video_path))
#                 create_objects_file(folder)  # Create objects.txt in each folder

# if __name__ == "__main__":
#     process_all_folders()
#     for i in range(20):
#         folder_path = Path(f"data/recordings/sid/sample_{i}")
#         if folder_path.exists():
#             rgb_path = folder_path / 'rgb'
#             if rgb_path.exists():
#                 output_path = folder_path / 'sid_video.mp4'
#                 create_video_from_images(str(rgb_path), str(output_path))
#             create_objects_file(folder_path)

# #-------------------------------------------------------------------------------------------#
# from BasicAI.functions.dataset.augmentation import DataAugmenter
# def augment_processed_predictions(action_name="sprinkle", num_samples=20):
#     base_path = Path("data/recordings")
#     action_path = base_path / action_name
    
#     for i in range(num_samples):
#         folder_path = action_path / f"sample_{i}"
#         input_csv = folder_path / "processed_predictions_hamer.csv"
#         output_csv = folder_path / "augmented_predictions.csv"
        
#         if input_csv.exists():
#             augmenter = DataAugmenter(str(input_csv), str(output_csv))
#             augmenter.augment_data()
#             augmenter.save_to_csv()
#             print(f"Augmented data saved to {output_csv}")

# if __name__ == "__main__":
#     augment_processed_predictions("sid")
#----------------------------------------------------------------------------------------------#
# def rename_videos(base_path="data/recordings/honey_bread"):
#     base_dir = Path(base_path)
#     for folder in base_dir.iterdir():
#         if folder.is_dir():
#             old_video_path = folder / 'honey_bread_video.mp4'
#             if old_video_path.exists():
#                 new_video_path = folder / 'ketchup_video.mp4'
#                 old_video_path.rename(new_video_path)
#                 print(f"Renamed {old_video_path} to {new_video_path}")

# if __name__ == "__main__":
#     rename_videos()
# # #-------------------------------------------------------------------------------------------#
# from Database.functions.rlef import RLEFManager
# from Config.config import load_config
# import http

# def upload_data_to_rlef(action_name="sprinkle"):
#     config = load_config("db_config.yaml")['RLEF']
#     manager = RLEFManager(config)
#     base_path = Path("data/recordings") / action_name
#     num_samples = 0

#     for folder in base_path.iterdir():
#         if folder.is_dir():
#             video_path = folder / f'{action_name}_video.mp4'
#             print("-"*100)
#             if video_path.exists():
#                 print(f"Uploading video from {folder}")
#                 try:
#                     task_id = manager.get_or_create_task(action_name)
#                     print(f"Current_Task_ID: {task_id}")
#                     status_code = manager.upload_to_rlef(str(video_path), task_id)
                    
#                     if status_code == http.HTTPStatus.OK:
#                         print("Data sent successfully to RLEF")
#                         num_samples += 1
#                     else:
#                         print(f"Failed to upload to RLEF. Status code: {status_code}")
                        
#                 except Exception as e:
#                     print(f"An error occurred while sending data to RLEF: {e}")

#     status = ("Ready for learning" if num_samples >= 30 
#              else f"Insufficient samples - need at least 30 (current: {num_samples})")
#     print(status)

# if __name__ == "__main__":
#     upload_data_to_rlef("ketchup")
# -------------------------------------------------------------------------------------------------#
def combine_augmented_csvs(action_name="ketchup", num_samples=21):
    base_path = Path("data/recordings") / action_name
    combined_csv_path = base_path / "dataset_non_aug.csv"
    combined_df = pd.DataFrame()

    for i in range(num_samples):
        csv_path = base_path / f"sample_{i}" / "processed_predictions_hamer.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV saved to {combined_csv_path}")

if __name__ == "__main__":
    combine_augmented_csvs("ketchup")