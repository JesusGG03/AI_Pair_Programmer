import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import configuration and routes
from .config import CONFIG
from .routes import assistant

# --- Logging Setup ---
# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(name)s - %(levelname)-s - %(message)s'
)
logger = logging.getLogger("ai_pair_programmer")
logger.info("Initializing FastAPI Application...")


# --- Application Factory Function ---
def create_app() -> FastAPI:
    ##Creates and configures the FastAPI application instance
    
    app = FastAPI(
        title=CONFIG.APP_NAME,
        description=CONFIG.APP_DESCRIPTION,
        version=CONFIG.API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- CORS Middleware (Crucial for IDE Plugin communication) ---
    # Since your IDE plugin may run on a different port/origin than the server,
    # we need to enable Cross-Origin Resource Sharing (CORS).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for local development/testing
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Route Inclusion ---
    # Include the assistant router to handle the /api/v1/query endpoint
    app.include_router(assistant.router)

    # --- Root Health Check Endpoint ---
    @app.get("/", summary="Health Check")
    def read_root():
        """Simple health check endpoint to verify the service is running."""
        return {"status": "ok", "service": CONFIG.APP_NAME, "version": CONFIG.API_VERSION}

    logger.info("Application setup complete.")
    return app

# Instantiate the application
app = create_app()
