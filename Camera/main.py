"""
This is the enter point of the camera microservice.
"""
import os
import sys
import uvicorn
from fastapi import FastAPI, Response
from routers.router import camera_router

# ---------------------------------------#
app = FastAPI(title="Camera Microservice", description="AI Hand IntelRealSense Camera Microservice", version="0.1.0", docs_url="/docs", redoc_url=None)
app.include_router(camera_router)
# ---------------------------------------#

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)