import logging #For event logging
import numpy as np #This will be used for the recording
import sounddevice as sd #This will talk to the hardware
import whisper #STT model
from scipy.io.wavfile import write as write_wav #This will write the wav file
import asyncio
import os
import tempfile
 

# Config logger
logger = logging.getLogger(__name__)

try:
    from ..config import CONFIG
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from app.config import CONFIG


# ---- VOICE ACTIVATION DETECTION (VAD) settings ----
# These settings control how the recorder knows when you're speaking
# For now, these will be hard coded. In the future, users will be able to modify these settings throught the request

# How loud the audio needs to be to be considered "speech"
VAD_THRESHOLD = .01

# How many seconds of silence to wait before stopping
VAD_SILENCE_DURATION = 2 # Two seconds

class SSTService:
    """
    A dedicated client for handling Speech-to-Text (STT) operations.
    It records audio from the microphone and uses Whisper to transcribe it.
    """
    def __init__(self):
        """
        Initializes the STTService.
        
        This loads the Whisper model into memory.
        """

        # Assigning the mdoel and sample rate from config
        self.model_name = CONFIG.WHISPER_MODEL
        self.samplerate = CONFIG.AUDIO_SAMPLE_RATE

        # Chunk sizes for VAD
        self.chunk_duration = .5 # 500ms Chunk
        self.chunk_size = int(self.samplerate * self.chunk_duration) # 8000 samples

        # Calculations for silent chunks needed
        self.silent_chunks_needed = int(VAD_SILENCE_DURATION / self.chunk_duration)

        # Loading STT model
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except:
            logger.error(f"Failed to load whisper model: {e}")
            self.model = None


    def _record_audio_with_vad(self) -> np.ndarray | None:
        """
        Records audio from the default microphone with simple VAD.
        
        This is a BLOCKING function that runs until silence is detected.
        It should be called from an async function using `asyncio.to_thread`.
        
        Returns:
            A NumPy array of the recorded audio, or None if nothing was recorded.
        """
        pass

    def _transcribe_audio_data(self, audio_data: np.ndarray) -> str:
        """
        Transcribes a NumPy array of audio data using Whisper.
        
        This is a BLOCKING, CPU/GPU-bound function.
        It should be called from an async function using `asyncio.to_thread`.
        
        Args:
            audio_data: The NumPy array of audio to transcribe.
        
        Returns:
            The transcribed text.
        """
        pass

    async def listen_and_transcribe(self) -> str:
        """
        The main public async method for this service.
        
        It runs the blocking record and transcribe functions in separate
        threads to avoid blocking the main server event loop.
        
        Returns:
            The transcribed text as a string.
        """
        pass    
