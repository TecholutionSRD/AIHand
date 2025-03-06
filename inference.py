import asyncio
from PIL import Image
from typing import List, Dict, Optional
from VisionAI.functions.vision_detection import GeminiInference
from Camera.functions.camera_reciver import CameraReceiver
from Config.config import load_config

# async def detect_object_centers(camera, object_names: List[str], config) -> List[Dict]:
#     """
#     Detect objects and their centers using Gemini model.
    
#     Args:
#         camera: CameraReceiver instance
#         object_names (List[str]): List of object names to detect
#         config (Dict): Configuration dictionary
        
#     Returns:
#         List[Dict]: List of dictionaries containing object centers and bounding boxes
#     """
#     # Initialize Gemini detector
#     detector = GeminiInference(config)
    
#     results = []
    
#     # Process each object name
#     for object_name in object_names:
#         try:
#             # Get object center and bounding box
#             detection_result = await detector.get_object_center(camera, object_name)
            
#             if detection_result:
#                 result = {
#                     "object_name": object_name,
#                     "center": detection_result["center"],
#                     "bounding_box": detection_result["box"],
#                     "confidence": detection_result["confidence"]
#                 }
#                 results.append(result)
#                 print(f"Detected {object_name} at center {detection_result['center']}")
#             else:
#                 print(f"Could not detect {object_name}")
                
#         except Exception as e:
#             print(f"Error detecting {object_name}: {e}")
#             continue
            
#     return results

async def main():
        config = load_config("vision_ai_config.yaml")
        camera = CameraReceiver('camera_config.yaml')
        detector = GeminiInference(config)
        # object_names = ["bottle" , "bread"]
        outputs = await detector.detect(camera, "bottle")
        outputs1 = await detector.detect(camera, "bread")
        print("bottle", outputs)
        print("bread", outputs1)

if __name__ == "__main__":
    asyncio.run(main())