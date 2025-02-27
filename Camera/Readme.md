# Camera Microservice

## Overview

This microservice provides functionalities for camera operations, including capturing images, recording videos, and streaming camera feeds. It is designed for modularity and scalability, supporting both hardware cameras and streamed inputs.

## Features

- **Image Capture:** Captures still images from connected cameras.
- **Video Recording:** Records video streams with customizable FPS and duration.
- **Live Streaming:** Streams live camera feeds over a network.
- **Camera Management:** Manages multiple camera connections and configurations.
- **API Endpoints:** Provides FastAPI-based endpoints for controlling camera functions.
- **Asynchronous Processing:** Uses `asyncio` for efficient frame handling.
- **Configurable Storage:** Saves recordings and frames in structured directories.

## Technologies Used

- **Programming Language:** Python
- **Framework:** FastAPI
- **Camera Library:** OpenCV
- **Streaming Protocol:** RTSP/WebRTC
- **Asynchronous Processing:** `asyncio`
- **Data Handling:** NumPy, PIL

## Setup Instructions

### **Prerequisites**
- Python 3.8+
- OpenCV (`opencv-python`)
- FastAPI (`fastapi`, `uvicorn`)
- Dependencies listed in `requirements.txt`

### **Installation**
Install required dependencies:
```bash
pip install -r requirements.txt
```

### **Configuration**
- Modify camera settings in `Camera/config/camera_config.yaml`.
- Set video parameters (FPS, resolution, format) in the configuration.

### **Running the Service**
Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### **Camera Operations**
- `GET /camera/health` → Checks if the camera service is running.
- `GET /camera/status` → Returns the active camera type (Hardware/Stream).
- `GET /camera/start` → Starts the camera or receiver.
- `GET /camera/stop` → Stops the camera or receiver.
- `GET /camera/capture_frame` → Captures and returns an RGB & depth frame.

### **Video Recording**
- `POST /camera/record` → Starts a video recording with configurable duration, FPS, and action labels.

## Folder Structure

```
Camera/
│── config/
│   ├── camera_config.yaml  # Configuration settings
│── utils/
│   ├── camera.py           # Hardware camera handler
│   ├── camera_receiver.py  # Stream receiver
│── recordings/
│   ├── [action_name]/sample_[n]/rgb/   # RGB Frames
│   ├── [action_name]/sample_[n]/depth/ # Depth Frames
│   ├── [action_name]/sample_[n].mp4    # Video File
```

## Contributing

Contributions are welcome! If you'd like to add features or fix issues, please submit a pull request.

## License

[MIT](LICENSE)
