from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os
import sys

# Add project root to sys.path to allow importing from root config
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from app.api.routes import prediction, measurement, health, simulation

app = FastAPI(title="Load Profile API")

# Setup CORS if needed
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router, prefix="/predictions", tags=["Predictions"])
app.include_router(measurement.router, prefix="/measurements", tags=["Measurements"])
app.include_router(simulation.router, prefix="/simulation", tags=["Simulation"])
app.include_router(health.router, tags=["Health"])

# Serve static UI files
ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ui"))
if not os.path.exists(ui_dir):
    os.makedirs(ui_dir)
app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

@app.get("/")
def redirect_to_ui():
    return RedirectResponse(url="/ui/")