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

class STTService:
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
        except Exception as e:
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

        audio_frames = []
        silent_chunks = 0
        is_speaking = False

        try:
            with sd.InputStream(
                samplerate=self.samplerate,
                blocksize=self.chunk_size,
                channels=1,
                dtype='float32'
            ) as stream:
                logger.info("Listening for speech... (speak now)")

                while True:
                    # Reads a chunk of audio
                    frame, overflowed = stream.read(self.chunk_size)
                    if overflowed:
                        logger.warning("Audio buffer overflowed!")

                    # Calculate the energy (RMS) ofa chunk
                    rms = np.sqrt(np.mean(frame**2))

                    if rms > VAD_THRESHOLD:
                        # Speech detected
                        if not is_speaking:
                            logger.info("Speech detected! Recording...")
                            is_speaking = True
                        # Setting to zero so the silent chunks dont accumalate if the user stops for a brief moment
                        silent_chunks = 0
                        # Append the array of audio to the audio frames
                        audio_frames.append(frame)
                    
                    elif is_speaking:
                        silent_chunks += 1
                        audio_frames.append(frame)

                        if silent_chunks > self.silent_chunks_needed:
                            logger.info(f"Silence detected for {VAD_SILENCE_DURATION}s, stopping recording.")
                            break
                    # Else, continue looping.

                if not audio_frames:
                    logger.warning("No audio recorded.")
                    return None
                
                audio_data = np.concatenate(audio_frames, axis=0)
                return audio_data 
            
        except sd.PortAudioError as e:
            logger.error(f"PortAudio error: {e}")
            logger.error("No Microphone in use or found")
            return None

        except Exception as e:
            logger.error(f"An error occured while recording: {e}")
            return None

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

        # Checks to see if model was able to be laoded from init
        if self.model is None:
            logger.error("Whisper model not loaded, cannot transcribe.")
            return "Error: Whisper model not loaded."
        
        # 1. Save audio data to a temp WAV file
        # delete set to false so we can control when this file is deleted
        with tempfile.TemporaryFile(delete=False, suffix=".wav") as tmp_file:
            write_wav(tmp_file.name, self.samplerate, audio_data)
            tmp_file_path = tmp_file.name

        logger.info(f"Audio saved to temp file: {tmp_file_path}")

        # 2. Transcribe the file
        try:
            result = self.model.transcribe(tmp_file_path, fp16=False)
            # Get the text from the returning dict
            transcription = result.get('text', '').strip()
            logger.info(f"Transcription complete: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return f"Error: {e}"
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                logger.debug(f"Temp file {tmp_file_path} deleted.")

    async def listen_and_transcribe(self) -> str:
        """
        The main public async method for this service.
        
        It runs the blocking record and transcribe functions in separate
        threads to avoid blocking the main server event loop.
        
        Returns:
            The transcribed text as a string.
        """

        if self.model is None:
            return "Error: STTService not initialized."
        
        # 1. Record audio
        logger.info("Switching to thread for audio recording...")
        audio_data = await asyncio.to_thread(self._record_audio_with_vad)

        if audio_data is None:
            return "Error: No audio detected or recording failed."
        

        # 2. Transcribe audio
        logger.info("Audio recorded, switching to thread for transcription...")
        transcription = await asyncio.to_thread(self._transcribe_audio_data, audio_data)

        return transcription
        
# --- Independent Test Block ---
# This special block allows you to run this file directly to test it:
# From your project's root folder, run:
# python -m app.infrastructure.stt_handler
#
if __name__ == "__main__":
    
    # We need to configure logging to see the output for testing
    logging.basicConfig(
        level=logging.INFO, 
        format='%(name)s - %(levelname)-s - %(message)s'
    )
    
    async def main_test():
        print("--- Testing STTService ---")
        try:
            service = STTService()
            if service.model is None:
                raise Exception("Whisper model failed to load.")
                
            transcription = await service.listen_and_transcribe()
            
            print(f"\n--- SUCCESS ---")
            print(f"Final Transcription: {transcription}")
            
        except Exception as e:
            print(f"\n--- FAILED ---")
            print(f"Test failed: {e}")
            print("\n--- Troubleshooting ---")
            print("1. Do you have a microphone connected and enabled?")
            print("2. Did you install all dependencies? `pip install openai-whisper sounddevice scipy numpy`")
            print("3. (Linux/macOS) Did you install PortAudio? `brew install portaudio` or `sudo apt-get install portaudio19-dev`")

    # Run the async test function
    asyncio.run(main_test())
