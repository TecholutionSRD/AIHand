"""
This is the entry point of the AI Hand Backend microservices
"""
import os
import sys
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from Camera.routers.router import camera_router
from VisionAI.routers.router import visionai_router
from Database.routers.router import db_router
from Database.routers.grasp_routes import grasp_router
from Database.routers.action_routes import action_router
from BasicAI.routers.router import ai_router

# ---------------------------------------#
# Create the FastAPI app
app = FastAPI(
    title="AI Hand", 
    description="Microservices for AI Hand",
    version="0.1.0", 
    docs_url="/ddocs",
    redoc_url="/rdocs",
    openapi_tags=[
        {"name": "Camera", "description": "Camera control operations including starting/stopping cameras, capturing frames, and recording videos."},
        {"name": "Vision", "description": "Computer vision operations including object detection, center point detection, and real-world coordinate mapping."},
        {"name": "AI", "description": "AI operations including data preprocessing, model training, real-time inference, and trajectory prediction."},
        {"name": "DataBase", "description": "Core database operations including connection management, database creation, and collection management."},
        {"name": "Action DB", "description": "CRUD for managing action records in database including creating, reading, updating, and deleting actions."},
        {"name": "Grasp DB", "description": "CRUD for managing grasp records in database including creating, reading, updating, and deleting grasp."}
    ]
)

# ---------------------------------------#
# Custom Docs
css_path = "static/custom_swagger.css"
html_path = "static/custom_docs.html"

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def custom_docs():
    with open(html_path) as f:
        return f.read()
# ---------------------------------------#
# CROS Allow
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

# ---------------------------------------#
# Add Routes
app.include_router(camera_router, tags=["Camera"],)
app.include_router(visionai_router, tags=["Vision"])
app.include_router(ai_router, tags=["AI"])
app.include_router(db_router, tags=["DataBase"])
app.include_router(action_router, tags=["Action DB"])
app.include_router(grasp_router, tags=["Grasp DB"])

# ---------------------------------------#
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)