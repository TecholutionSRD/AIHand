
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
from Camera.utils.camera_reciver import CameraReceiver

class VideoRecorder:
    """
    Class to record an 8-second video at 30 FPS and save frames at 2 FPS,
    while also capturing initial RGB and depth images before each recording.
    """
    def __init__(self, receiver, config, num_recordings, action_name, objects):
        """
        Initializes the VideoRecorder with the provided CameraReceiver instance and additional parameters.
        """
        self.config = config.get("Video_Recorder", {})
        self.receiver = receiver
        self.output_dir = self.config['data_path']
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
        sample_folder = os.path.join(self.output_dir, self.action_name, f"sample_{self.sample_count + 1}")
        sub_dirs = ["rgb", "depth", "initial_frames"]

        # Create all required directories
        for sub_dir in ["", *sub_dirs]:  
            os.makedirs(os.path.join(sample_folder, sub_dir), exist_ok=True)

        return sample_folder

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

            await self.capture_initial_frames(sample_folder)
            
            print("")
            for i in range(5, 0, -1):
                print(f"\r[Video Recorder] Countdown: {i} seconds remaining", end="")
                await asyncio.sleep(1)

            video_path = os.path.join(sample_folder, f"{self.action_name}_video.{self.config['video_format']}")
            print("")
            print(f"[Video Recorder] Video will be saved to: {video_path}")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_size = (self.config['width'], self.config['height'])
            out = cv2.VideoWriter(video_path, fourcc, self.config['video_fps'], frame_size)

            if not out.isOpened():
                print("[Video Recorder] ERROR: VideoWriter failed to open")
                return

            start_time = time.time()
            frame_time = 1.0 / self.config['video_fps']
            total_frames = int(self.config['video_duration'] * self.config['video_fps'])

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
