import asyncio

from app.helpers.video_downloader import RutubeDownloader
from app.helpers.video_preprocessing import AudioExtractor, AudioTranscriber


def main():
    downloader = RutubeDownloader()
    transcriber = AudioTranscriber('turbo')

    path_to_video = asyncio.run(downloader.download_file(input()))
    with AudioExtractor() as extractor:
        path_to_audio = extractor.extract_audio(path_to_video)
        text = transcriber.transcribe(path_to_audio)

    print(text)


if __name__ == '__main__':
    main()
