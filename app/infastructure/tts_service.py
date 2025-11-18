import logging
import asyncio
from typing import AsyncGenerator
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import io
import os
import tempfile
from TTS.api import TTS

# Use relative imports to get your config
try:
    from ..config import CONFIG
except ImportError:
    # Fallback for direct script execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from app.config import CONFIG

logger = logging.getLogger(__name__)


# Punctuation
SENTENCE_ENDINGS = [".", "!", "?", "\n"]


class TTSService:
    """
    A dedicated client for handling Text-to-Speech (TTS) operations.
    
    This service now runs LOCALLY in the main app, using the
    `coqui-tts` forked library.
    """
    def __init__(self):
        """
        Initializes the TTSService by loading the Coqui TTS model.
        This is a SLOW, BLOCKING operation.
        """
        self.model_name = CONFIG.TTS_MODEL_NAME
        self.speaker = CONFIG.TTS_VOICE_NAME
        self.sample_rate = None # Will be provided by the model

        try:
            logger.info(f"Loading TTS model: {self.model_name}")
            logger.info("This may take a moment, especially on first run...")
            # Loading this model. This is a blocking call
            self.model = TTS(model_name=self.model_name, progress_bar=True)

            # Get the sample rate from the model itself
            self.sample_rate = self.model.synthesizer.output_sample_rate
            logger.info("TTS model loaded successfully.")

            # Pre warm the model. That way its not slow on the first speak
            self.model.tts(text="Warmup", speaker=self.speaker)
            logger.info(f"TTS model pre-warmed. Sample rate: {self.sample_rate}Hz")

        except Exception as e:
            logger.critical(f"FATAL: Could not load TTS model: {e}")
            logger.critical("Please ensure TTS is installed (`pip install coqui-tts`)")
            self.model = None
        except KeyboardInterrupt:
            logger.info("TTS model loading cancelled.")
            self.model = None

    def _generate_audio_blocking(self, text: str) -> np.ndarray | None:
        """
        Generates audio data from text using the loaded TTS model.
        
        This is a BLOCKING, CPU/GPU-bound function.
        It should be called from an async function using `asyncio.to_thread`.
        
        Args:
            text: The text to synthesize.
        
        Returns:
            A NumPy array of the audio data, or None if generation failed.
        """
        if self.model is None:
            logger.error("Cannot synthesize, TTS model is not loaded.")
            return None
        
        if not text or not text.strip():
            return None
        
        # We are generating the audio using TTS. After that, we are passing the converting it into the audio data
        try:
            # This is a Blocking operations. 
            # It returns a lists of floating values
            audio_list = self.model.tts(text=text, speaker=self.speaker)

            # Convert it into an array for playback
            audio_data = np.array(audio_list, dtype='float32')

            # Silence for a more smooth sentence transition
            silence_duration = .25
            silence_sample = int(self.sample_rate * silence_duration)
            silent_chunks = np.zeros(silence_sample, dtype='float32')

            # Append the silent chunks to the data
            padded_audio = np.concatenate((audio_data, silent_chunks))
            return padded_audio
        except Exception as e:
            logger.error(f"TTS generation failed for text '{text}': {e}")
            return None

    def _play_audio_blocking(self, audio_data: np.ndarray):
        """
        Plays a NumPy array of audio data.
        
        This is a BLOCKING function that waits until playback is finished.

        Args:
            audio_data: The array of float values to play
        
        """
        if self.sample_rate is None:
            logger.error("Cannot play audio: sample rate unknown.")
            return
        
        try:
            sd.play(audio_data, self.sample_rate, blocking=True)
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
        
    async def _audio_playback_consumer(self, audio_queue: asyncio.Queue):
        """
        This async function runs in the background, pulling audio chunks
        from the queue and playing them sequentially in a worker thread.

        Args:
            audio_queue: The text to synthesize.
        
        """
        logger.info("Audio consumer started, waiting for audio chunks...")
        while True:
            # 1. Get Audio chunk to play
            audio_chunk = await audio_queue.get()

            # 2. Check if audio chunk is present. If not, end Queue and break
            if audio_chunk is None:
                logger.info("Stop signal received, audio consumer shutting down.")
                audio_queue.task_done()
                break
            
            # 3. Since chunk is present, send it and play it in another thread. This prevents it from blocking
            logger.debug("Got audio chunk, playing in thread...")
            await asyncio.to_thread(self._play_audio_blocking, audio_chunk)
            audio_queue.task_done()
    
    def _split_buffer(self, buffer: str) -> (str | None, str):
        """
        Splits a text buffer into the first complete sentence and the rest.

        Args:
            buffer:  Text to be split

        Returns:
            A String
        """
        for ending in SENTENCE_ENDINGS:
            if ending in buffer:
                parts = buffer.split(ending, 1)
                sentence = parts[0] + ending
                remaining_buffer = parts[1]

                return sentence.strip(), remaining_buffer.strip()
        return None, buffer

    async def speak_text_stream(self, text_generator: AsyncGenerator[str, None]):
        """
        The main public async method for this service.
        
        This "Producer" function consumes the LLM's text stream,
        generates audio for each sentence, and puts audio chunks
        in a queue for the "Consumer" to play.
        """
        if self.model is None:
            logger.error("Cannot speak, TTS model is not loaded.")
            return
        
        audio_queue = asyncio.Queue()
        # Set up the playback in the back ground
        consumer_task = asyncio.create_task(self._audio_playback_consumer(audio_queue))

        text_buffer = ""

        try:
            async for text_chunk in text_generator:

                # 1. Add the text_chunk to the buffer
                text_buffer += text_chunk

                # 2. Check to see if the text buffer has a full sentence
                sentence, remaining_buffer = self._split_buffer(text_buffer)

                # 3. While sentence is Not empty/null, loop and create audio and put it in the queue
                while sentence:
                    logger.info(f"Generating audio for: '{sentence}'")

                    # 3.1 Create audio chunk
                    audio_chunk = await asyncio.to_thread(self._generate_audio_blocking, sentence)

                    # 3.2 If audio chunk is not null, put it in the queue
                    if audio_chunk is not None:
                        await audio_queue.put(audio_chunk)

                    # 3.3 Check to see if there are any remaining sentence
                    # This way sentence is still being checked when looping with while
                    text_buffer = remaining_buffer
                    sentence, remaining_buffer = self._split_buffer(text_buffer)
                
            # 4 Check for any remaining buffer, check to see if not null, put in buffer
            if text_buffer.strip():
                logger.info(f'Checking for remaining buffer in: {text_buffer}')
                audio_chunk = await asyncio.to_thread(self._generate_audio_blocking, text_buffer)
                
                if audio_chunk is not None:
                    await audio_queue.put(audio_chunk)
        except Exception as e:
            logger.error(f"Error in text stream producer: {e}")
        finally:

            # 5. Send signal to stop and wait for queue to finish
            await audio_queue.put(None)

            await consumer_task
            logger.info("TTS stream complete.")
    
    async def close(self):
        """A placeholder function in case we need cleanup later."""
        # We don't need to close an HTTP client anymore
        logger.info("TTSService (local) closed.")
        pass

# --- Independent Test Block ---
if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(name)s - %(levelname)-s - %(message)s'
    )
    
    async def mock_text_generator():
        """A fake LLM text stream generator for testing."""
        text = "Hello! This is the first sentence for testing. Here is a second one! And finally, a third."
        
        for word in text.split(" "):
            yield f"{word} "
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(0.5)

    async def main_test():
        print("--- Testing TTSService (LOCAL `coqui-tts`) ---")
        
        service = None
        try:
            # 1. Initialize the service (THIS IS SLOW - LOADS MODEL)
            print("Loading TTS model...")
            service = TTSService()
            if service.model is None:
                raise Exception("TTS model failed to load.")
            
            # 2. Run the test
            print("\nStarting playback of mock text stream...")
            print("You should hear audio play in three chunks.")
            
            await service.speak_text_stream(mock_text_generator())
            
            print(f"\n--- SUCCESS ---")
            print(f"Playback complete.")
            
        except Exception as e:
            print(f"\n--- FAILED ---")
            print(f"Test failed: {e}")
            print("\n--- Troubleshooting ---")
            print("1. Did you install all dependencies? `pip install coqui-tts sounddevice scipy numpy`")
            print("2. Do you have a working internet connection (for the first model download)?")
            print("3. Do you have speakers/headphones connected?")
        finally:
            if service:
                await service.close()

    # Run the async test function
    asyncio.run(main_test())
