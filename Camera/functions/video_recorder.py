
"""
This file is responsible for recording n second videos and turning on and off the interactive grid.
"""
from pathlib import Path
from typing import List
import cv2
import os
import time
import asyncio
import numpy as np
import sys
import traceback
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Config.config import load_config
from Camera.functions.camera_receiver import CameraReceiver

class VideoRecorder:
    """
    Class to record an `n`-second video at `m` FPS and save frames at `2` FPS,
    while also capturing initial RGB and depth images before each recording.
    """
    def __init__(self, receiver:CameraReceiver, config_path:Path, num_recordings:int=1, action_name:str="pouring", objects:List[str]=["red soda can"]):
        """
        Initializes the VideoRecorder with the provided CameraReceiver instance and additional parameters.
        """
        config = load_config(config_path)
        self.config = config.get("Video_Recorder", {})
        self.receiver = receiver
        self.output_dir = self.config.get('data_path',"data/recordings/")
        self.num_recordings = num_recordings
        self.action_name = action_name
        self.objects = objects
        self.sample_count = len([d for d in os.listdir(f"{self.output_dir}/{self.action_name}") 
                    if os.path.isdir(f"{self.output_dir}/{self.action_name}/{d}") and d.startswith("sample_")]) if os.path.exists(f"{self.output_dir}/{self.action_name}") else 0

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[Video Recorder] Count : {self.sample_count}")
        self.has_display = os.environ.get('DISPLAY') is not None

    def create_sample_directory(self):
        """
        Creates a new sample directory structure under the action name.
        """
        sample_folder = Path(self.output_dir) / self.action_name / f"sample_{self.sample_count + 1}"
        
        # Create all required directories in one go
        sub_dirs = ["rgb", "depth", "initial_frames"]
        [sample_folder.joinpath(sub_dir).mkdir(parents=True, exist_ok=True) for sub_dir in sub_dirs]

        return str(sample_folder)

    async def capture_initial_frames(self, sample_folder):
        """
        Captures a single RGB and depth frame and saves them before video recording.
        """
        try:
            color_frame, depth_frame = await self.receiver.decode_frames()
            if color_frame is not None:
                cv2.imwrite(os.path.join(sample_folder, "initial_frames", "image_0.png"), color_frame)
                if self.has_display:
                    cv2.imshow("Initial RGB Frame", color_frame)
                    cv2.waitKey(1)
            if depth_frame is not None:
                np.save(os.path.join(sample_folder, "initial_frames", "image_0.npy"), depth_frame)

        except Exception as e:
            print(f"[Video Recorder] ERROR capturing initial frames: {traceback.format_exc()}")
            
    async def record_video(self):
        """
        Records videos based on the number of samples in the configuration.
        """
        print(f"[Video Recorder] Starting video recording with {self.num_recordings} recordings")

        for recording_num in range(self.num_recordings):
            print(f"[Video Recorder] Starting recording {recording_num + 1} of {self.num_recordings}")

            sample_folder = self.create_sample_directory()
            print(f"[Video Recorder] Created sample directory: {sample_folder}")

            with open(os.path.join(sample_folder, "objects.txt"), 'w') as f:
                for obj in self.objects:
                    f.write(f"{obj}\n")
            
            _, _ = await self.receiver.decode_frames()
            await asyncio.sleep(1)

            print("\n[Video Recorder] Starting recording now!")

            video_path = os.path.join(sample_folder, f"{self.action_name}_video.{self.config.get('video_format', 'mp4')}")
            print(f"[Video Recorder] Video will be saved to: {video_path}")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_size = (self.config.get('width', 640), self.config.get('height',480))
            out = cv2.VideoWriter(video_path, fourcc, self.config.get('video_fps',10), frame_size)

            if not out.isOpened():
                print("[Video Recorder] ERROR: VideoWriter failed to open")
                return

            # Recording phase starts here
            start_time = time.time()
            frame_time = 1.0 / self.config.get('video_fps',10)
            total_frames = int(self.config.get('video_duration',8) * self.config.get('video_fps',10))

            print(f"[Video Recorder] Recording {total_frames} frames")
            print("")
            for frame_count in range(total_frames):
                try:
                    color_frame, depth_frame = await self.receiver.decode_frames()
                    current_time = time.time()
                    delay = max(0, start_time + (frame_count * frame_time) - current_time)
                    await asyncio.sleep(delay)

                    if color_frame is not None:
                        color_frame = cv2.resize(color_frame, frame_size)
                        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                        out.write(color_frame)

                    frame_filename = f"{self.action_name}_image_{frame_count:03d}.png"
                    rgb_path = os.path.join(sample_folder, "rgb", frame_filename)
                    cv2.imwrite(rgb_path, color_frame)

                    if depth_frame is not None:
                        depth_path = os.path.join(sample_folder, "depth", f"{self.action_name}_image_{frame_count:03d}.npy")
                        np.save(depth_path, depth_frame)

                    print(f"\r[Video Recorder] Recorded frame {frame_count + 1}/{total_frames}", end="")

                except Exception as e:
                    print("")
                    print(f"[Video Recorder] ERROR during frame capture: {traceback.format_exc()}")
                    break

            out.release()
            print("")
            print(f"[Video Recorder] Recording completed - Video and frames saved in {sample_folder}")
            return video_path

    async def display_live_feed(self):
        """
        Displays the live feed from the receiver after recording completes.
        """
        print("[Video Recorder] Displaying live feed...")
        try:
            while True:
                color_frame, _ = await self.receiver.decode_frames()
                if color_frame is not None:
                    cv2.imshow("Live RGB Feed", color_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(f"[Video Recorder] ERROR displaying live feed: {traceback.format_exc()}")
        finally:
            cv2.destroyAllWindows()
