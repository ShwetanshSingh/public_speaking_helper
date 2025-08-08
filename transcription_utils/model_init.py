import os 

import whisper
from dotenv import load_dotenv

load_dotenv("./config/.env")

class WhisperModel:
    """Speech-to-text transcription using OpenAI's Whisper model.

    # Usage
    ```python
    model = WhisperModel()
    text = model.transcribe("path/to/audio.mp3")
    ```

    The model type can be configured via the `WHISPER_MODEL` environment variable.
    Default model is 'base' if not specified.
    """
    def __init__(self) -> None:
        self.model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))

    def transcribe(self, audio_file: str) -> str:
        """Convert audio to text using Whisper model.

        # Parameters
        * `audio_file` - Path to the audio file to transcribe

        # Returns
        * `str` - The transcribed text from the audio file

        # Raises
        * `Exception` - If model output is not a string type
        """
        result = self.model.transcribe(audio_file)
        transcription = result["text"]
        if not isinstance(transcription, str):
            raise Exception("Model output is not string, but list[Unknown]")

        return transcription
