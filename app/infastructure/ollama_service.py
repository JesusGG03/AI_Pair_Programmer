import ollama
import logging
import asyncio

from typing import AsyncGenerator

try:
    from ..config import CONFIG
    from ..core.models import IdeContext
except ImportError:
    # Fallback used for testing
    import sys
    import os
    # Add the parent directory (root) to the Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from app.config import CONFIG
    from app.core.models import IdeContext

logger = logging.getLogger(__name__)

# Prompt Template
PROMPT_TEMPLATE = """
You are a helpful, conversational AI pair programmer.
A user is asking a question about a file they have open in their IDE.
Be concise and clear in your answer. Do not use markdown or formatting.

---
FILE CONTEXT:
File Path: {file_path}
Selected Text: "{selected_text}"
Cursor Line: {cursor_line}

---
USER'S QUESTION:
"{transcription}"
---

Your answer:
"""

class OllamaService:
    """
    A dedicated client for communicating with the local Ollama service.
    """
    def __init__(self):
        """
        Initializes the OllamaService.
        
        It sets up the async client with the base URL and model name
        from the central configuration.
        """
        self.client = ollama.AsyncClient(host=CONFIG.OLLAMA_BASE_URL)
        self.model = CONFIG.LLM_MODEL_NAME
        logger.info(f"OllamaService initialized. Target: {CONFIG.OLLAMA_BASE_URL}, Model: {CONFIG.LLM_MODEL_NAME}")
    
    def _format_prompt(self, transcription: str, context: IdeContext) -> str:
        """
        Private helper function to build the final prompt string.
        
        Args:
            transcription: The text transcribed from the user's voice.
            context: The IdeContext object with file path, selection, etc.
        
        Returns:
            A formatted prompt string ready to be sent to the LLM.
        """
        return PROMPT_TEMPLATE.format(
            file_path=context.file_path,
            selected_text=context.selected_text or "None",
            cursor_line=context.cursor_line,
            transcription=transcription
        )

    async def get_llm_response(self, transcription: str, context: IdeContext) -> AsyncGenerator[str, None]:
        """
        Gets a response from the Ollama model based on the user's
        transcription and their current IDE context.
        
        This is the main public method for this class.
        
        Args:
            transcription: The text transcribed from the user's voice.
            context: The IdeContext object.
        
        Returns:
            A string containing the AI's plain-text response.
        
        Raises:
            Exception: If the Ollama API returns an error or is unreachable.
        """
        
        prompt = self._format_prompt(transcription, context)
        logger.info(f"Sending Prompt to Ollama Model: {self.model}")

        try:
            # generate function to talk to the LLM
            response_stream = await self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=True
            )

            # Becuase stream is true, it is an async generator
            # We must loop over it
            async for chunk in response_stream:
                # Each chunk is a dictionary. Extract text from response
                text_chunk = chunk.get('response', '')

                if text_chunk:
                    # Yield text_chunk back to user
                    yield text_chunk
            
        
        except ollama.ResponseError as e:
            logger.error(f"Ollama API returned an Error: {e.error}")
            raise Exception(f"Ollama API error: {e.error}")
        
        except:
            logger.error(f"Failed to connect to Ollama at {CONFIG.OLLAMA_BASE_URL}: {e}")
            raise Exception(f"Ollama connection failed. Is the server running?")

# --- Independent Test Block ---
# This special block allows you to run this file directly to test it:
# From your project's root folder, run:
# python -m app.infrastructure.ollama_client
#
# Make sure you have `pip install ollama` and `pip install pydantic`
# in your virtual environment.
if __name__ == "__main__":
    
    # We need to configure logging to see the output for testing
    logging.basicConfig(
        level=logging.INFO, 
        format='%(name)s - %(levelname)-s - %(message)s'
    )
    
    # A simple mock context for testing, using the Pydantic model
    mock_context = IdeContext(
        file_path="/test/example.py",
        file_content="def hello():\n  print('hello')",
        cursor_line=2,
        selected_text="print('hello')"
    )
    mock_transcription = "What does this line of code do?"

    # Create the service
    service = OllamaService()

    async def main_test():
        print("--- Testing OllamaService Client ---")
        print(f"Test Query: {mock_transcription}\n")
        print(f"AI Response: ")

        full_response = []
        try:
            # Loop through the async generator
            async for response_chunk in service.get_llm_response(mock_transcription, mock_context):
                # Print each chunk as it arrives
                print(response_chunk, end='', flush=True)
                full_response.append(response_chunk)

            print("\n\n--- SUCCESS (STREAMING) ---")
            logger.info("Full Response recieved")

        except Exception as e:
            print(f"\n--- FAILED ---")
            print(f"Test failed: {e}")
            print(f"Check that your Ollama server is running at: {CONFIG.OLLAMA_BASE_URL}")

    # Run the async test function
    asyncio.run(main_test())