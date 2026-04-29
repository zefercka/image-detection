from pathlib import Path
from audio_extract import extract_audio
import tempfile
import shutil

class AudioExtractor:
    def __init__(self):
        self._temp_dir = Path(tempfile.mkdtemp(prefix="audio_extract_"))

    def extract_audio(self, path_to_video: str | Path) -> str:
        path_to_video = Path(path_to_video)
        file_name = f"{path_to_video.stem}.mp3"
        
        temp_file = self._temp_dir / file_name

        extract_audio(input_path=str(path_to_video), output_path=str(temp_file))

        return str(temp_file.absolute())
    
    def __enter__(self) -> "AudioExtractor":
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        return False
