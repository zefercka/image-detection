import whisper
from pathlib import Path


class AudioTranscriber:
    def __init__(self, model: str):
        self._model = whisper.load_model(model)
        
    def transcribe(self, path_to_audio: Path | str):
        path_to_audio = str(path_to_audio)
        
        result = self._model.transcribe(path_to_audio)
        
        return result['text']
