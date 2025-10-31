import logging #For event logging
import io #To create in memory files
import numpy as np #This will be used for the recording
import sounddevice as sd #This will talk to the hardware
import whisper #STT model
from scipy.io.wavfile import write as write_wav #This will write the wav file

# Config logger
logger = logging.getLogger(__name__)


class stt_service:
    """
    Handles all Speech-to-Text (STT) operations.
    This class is responsible for:
    1.  Listening to the user's microphone.
    2.  Detecting when the user has finished speaking (Voice Activity Detection).
    3.  Transcribing the spoken audio to text using Whisper.
    """


    def __init__(self, model_size: str = "base.en", silence_threshold_db: int = 40, silence_duration_sec: int = 2):
        """
        Initializes the STTService.
        
        Args:
            model_size (str): The size of the Whisper model to load (e.g., "tiny.en", "base.en").
                              "base.en" is a good balance of speed and accuracy for English.
            silence_threshold_db (int): The audio level (in dB) below which audio is
                                        considered "silence".
            silence_duration_sec (int): How long (in seconds) the audio must be "silent"
                                        before stopping the recording.
        """
        
        logger.info(f"Loading Whisper model: {model_size}...")

        try:
            self.model = whisper.load_model(model_size)
            logger.info("Whisper gmodel loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load whisper Model '{model_size}'. Error: {e}", exc_info=True)
            raise

        self.samplerate = 16000 # kHz frequency for Whisper
        self.channels = 1

        # Voice Activity Detection Parameters (VAD)
        # Conver silence threshold (Db) to a linear amplitude scale
        self.silence_threshold = 10 ** (silence_threshold_db / 20)
        self.silence_duration = silence_duration_sec

        # Calculate how many chunks are silent
        self.chunks_per_second = 5 # Processing audio 1/5th (200ms)
        
        self.silent_chunks_needed = self.silence_duration * self.chunks_per_second