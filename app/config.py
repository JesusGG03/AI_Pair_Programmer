import os
from dotenv import load_dotenv

# Load environment variables from .env file (optional, but good practice)
load_dotenv()

class AppConfig:
    """Central configuration class for the AI Pair Programmer application."""
    
    # --- Server Settings ---
    API_VERSION = "v1"
    APP_NAME = "AI Pair Programmer Service"
    APP_DESCRIPTION = "Local conversational AI service for pair programming."
    SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
    SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))

    # --- LLM/Ollama Settings (Example Defaults) ---
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "codellama:7b") # Use a fast, local model for testing

    # --- Audio Settings (STT/TTS) ---
    # Placeholder for configuration related to Whisper and Coqui
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
    AUDIO_SAMPLE_RATE = 16000 # Standard for speech processing
    TTS_VOICE_NAME = os.getenv("TTS_VOICE_NAME", "default_voice")


# Instantiate the configuration
CONFIG = AppConfig()

if __name__ == "__main__":
    # Simple check to ensure configuration loads correctly
    print(f"--- Loaded Configuration ---")
    print(f"API Name: {CONFIG.APP_NAME}")
    print(f"Ollama URL: {CONFIG.OLLAMA_BASE_URL}")
    print(f"Server Port: {CONFIG.SERVER_PORT}")
