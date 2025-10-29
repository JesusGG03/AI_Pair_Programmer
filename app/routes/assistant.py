import logging
import time

from fastapi import APIRouter
from fastapi.responses import Response


from ..core.models import IdeContext, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["pair-programmer"],
)


## mock agent. this will be done in /core/pair_agent and then initailized in main
class MockPairAgent:
    def process_query(self, context: IdeContext) -> dict:
        """
        Simulates the full STT -> LLM -> TTS -> Playback loop.
        Returns a dictionary with the text transcription and AI response.
        """
        logger.info(f"Context received for file: {context.file_path}")
        
        # 1. Simulate STT (Listening)
        logger.info("Starting STT (listening for user voice)...")
        # In a real app, this would block and wait for mic input
        time.sleep(1.5) # Simulate user speaking
        mock_transcription = "User said: 'Please explain this function to me.'"
        logger.info(f"Transcription complete: {mock_transcription}")

        # 2. Simulate LLM (Reasoning)
        logger.info("Sending context and transcription to LLM...")
        time.sleep(1) # Simulate LLM processing
        mock_response_text = "Hello! This function appears to initialize the database connection. It's crucial for..."
        logger.info(f"LLM response received: {mock_response_text}")

        # 3. Simulate TTS (Generating Audio)
        logger.info("Generating TTS audio from LLM response...")
        time.sleep(0.5) # Simulate audio generation

        # 4. Simulate Playback (The new, crucial step)
        logger.info("Playing audio response on local speakers...")
        # This is where you would use a library like 'playsound' or 'sounddevice'
        # with the generated audio bytes.
        time.sleep(2) # Simulate the audio file playing
        logger.info("Audio playback complete.")

        # 5. Return the text data for the JSON response
        return {
            "text_transcribed": mock_transcription,
            "ai_response_text": mock_response_text
        }

agent = MockPairAgent()

## Post request to start listening
@router.post("/query/",
             summary="Post request that handles AI Pair Programmer",
             response_model=QueryResponse,
             status_code=200)
async def handle_pair_query(context: IdeContext) -> QueryResponse:
    """
    Receives IDE context data, triggers the full S-T-S loop (including
    local audio playback), and returns a JSON status response.
    """
    logger.info("POST /api/v1/query received.")

    # 1. Pass the context to the PairAgent.
    # This call will block until the *entire* process,
    # including audio playback, is complete.
    try:
        response_data = agent.process_query(context)

        # 2. Return the JSON response to the IDE plugin
        return QueryResponse(
            status="success",
            text_transcribed=response_data["text_transcribed"],
            ai_response_text=response_data["ai_response_text"]
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        # Return a JSON response that indicates an error
        # Note: This still returns a 200 OK, but the body tells
        # the plugin that the operation failed.
        return QueryResponse(
            status="error",
            text_transcribed="N/A",
            ai_response_text="N/A",
            error_message=str(e)
        )