# Camera Microservice

## Overview

This microservice provides functionalities related to camera operations, such as capturing images, recording videos, and streaming camera feeds. It is designed to be a modular and scalable component within a larger system.

## Features

*   **Image Capture:** Captures still images from connected cameras.
*   **Video Recording:** Records video streams from connected cameras.
*   **Live Streaming:** Streams live camera feeds over a network.
*   **Camera Management:** Manages multiple camera connections and configurations.
*   **API Endpoints:** Provides API endpoints for controlling camera functions.

## Technologies Used

*   Programming Language: Python
*   Framework: Flask
*   Camera Library: OpenCV
*   Streaming Protocol: RTSP

## Setup Instructions

1.  **Prerequisites:**
    *   Python 3.6+
    *   OpenCV library
    *   Flask framework

2.  **Installation:**

    ```bash
    pip install flask opencv-python
    ```

3.  **Configuration:**
    *   Configure camera settings in the `config.py` file.
    *   Set the appropriate camera index or URL.

4.  **Running the Service:**

    ```bash
    python app.py
    ```

## API Endpoints

*   `/capture`: Captures an image from the default camera.
*   `/record`: Starts recording video from the default camera.
*   `/stream`: Streams live video from the default camera.

## Contributing

Contributions are welcome! Please submit a pull request with your proposed changes.

## License

[MIT](LICENSE)
