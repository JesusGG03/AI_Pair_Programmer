import uvicorn
from app.main import app
from app.config import CONFIG

if __name__ == "__main__":

    APP_MODULE_PATH = "app.main:app"


    # Runs the FastAPI application using uvicorn server
    print(f"--- Starting AI Pair Programmer Service ---")
    print(f"Access API Docs at http://{CONFIG.SERVER_HOST}:{CONFIG.SERVER_PORT}/docs")
    
    # We use the reload=True for development to automatically restart the server on code changes
    uvicorn.run(
        APP_MODULE_PATH, 
        host=CONFIG.SERVER_HOST, 
        port=CONFIG.SERVER_PORT, 
        log_level="info", 
        reload=True
    )
