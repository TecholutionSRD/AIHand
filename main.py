"""
This is the enter point of the camera microservice.
"""
import os
import sys
import uvicorn
from fastapi import FastAPI, Response
from Camera.routers.router import camera_router
from VisionAI.routers.router import visionai_router

# ---------------------------------------#
app = FastAPI(title="Camera Microservice", description="AI Hand IntelRealSense Camera Microservice", version="0.1.0", docs_url="/docs", redoc_url=None)
# ---------------------------------------#

# ---------------------------------------#
# Add Routes
app.include_router(camera_router, tags=["Camera"])
app.include_router(visionai_router, tags=["Vision AI"])
# ---------------------------------------#

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)