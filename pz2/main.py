import uuid
from pathlib import Path
from urllib.parse import urlencode

import cv2
import magic
import requests
import yt_dlp


SAVE_PATH = './frames'


class VideoDownloader:
    def __init__(self, base_url: str, save_path: str = './'):
        self._base_url = base_url
        self._save_path = Path(save_path.rstrip('/'))
    
    def download_file(self, url: str, resolution: int | None = None) -> str:
        raise NotImplementedError


class YandexDiskDownloader(VideoDownloader):
    def __init__(self,
                 base_url: str = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?',
                 save_path: str = './'):
        super().__init__(base_url, save_path)

    def _get_download_link(self, url: str):
        get_url = self._base_url + urlencode(dict(public_key=url))
        response = requests.get(get_url)

        return response.json()['href']

    def _write_file_on_disk(self, response, file_path: str):
        with open(file_path, 'wb') as f:
            f.write(response.content)

    def download_file(self, url: str, resolution: int | None = None) -> str:
        download_link = self._get_download_link(url)
        download_response = requests.get(download_link)

        filename = self._get_file_name(download_response.content)
        file_path = self._save_path / filename

        self._write_file_on_disk(download_response, file_path)
        
        return str(file_path.absolute())

    def _get_file_name(self, content: bytes):
        file_extension = magic.from_buffer(content, mime=True).split('/')[-1]
        print(magic.from_buffer(content, mime=True))
        return f"{uuid.uuid4()}.{file_extension}"


class RutubeDownloader(VideoDownloader):
    def __init__(self, save_path: str = './'):
        super().__init__('', save_path)
    
    def download_file(self, url: str, resolution: int | None = None) -> str:
        ydl_opts = {
            'outtmpl': str(self._save_path / '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
        }
        if resolution:
            ydl_opts['format'] = f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]'

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)
            file_path = Path(filename).with_suffix('.mp4')

            if file_path.exists():
                if input('Видео уже скачано. Скачать заново? ') == 'n':
                    return str(file_path.absolute())

            print('Video is downloading...')
            ydl.download([url])

        return str(file_path.absolute())
    
    
def split_to_frames(file_path: str, save_path: str, frame_interval: int):
    dir_name = file_path.split('/')[-1].split('.')[0]
    _save_path = Path(save_path) / dir_name
    
    if _save_path.exists():
        print(f'[WARNING] {str(_save_path.absolute())} already exists and will be recreated')
        _save_path.rmdir()

    _save_path.mkdir(parents=True)

    cap = cv2.VideoCapture(file_path)
    
    frame_counter = 0
    frame_index = 1
    while (r := cap.read())[0]:
        frame_counter += 1
        frame = r[1]
        
        if frame_counter % frame_interval == 0:
            frame_save_path = _save_path / f'frame_{frame_index}.jpg'
           
            success, encoded_img = cv2.imencode('.jpg', frame)
            
            if success:
                with open(frame_save_path, 'wb') as f:
                    f.write(encoded_img)

            frame_counter = 0
            frame_index += 1
    
    cap.release()

    print(f'[INFO] Всего сохранено {frame_index - 1} кадров')

    

url = input('Enter URL for download: ')

url_splitted = url.split('://')
if len(url_splitted) < 2:
    print("Invalid URL format")
else:
    if url_splitted[1].startswith('rutube'):
        downloader = RutubeDownloader()
        file_path = downloader.download_file(url)
        print(f"Downloaded file: {file_path}")
    elif url_splitted[1].startswith('disk.yandex.ru'):
        disk = YandexDiskDownloader()
        file_path = disk.download_file(url)
        print(f"Downloaded file: {file_path}")
    else:
        print("Unsupported URL format")
        exit(1)


frame_interval = int(input("Enter frame interval: "))
split_to_frames(file_path, save_path=SAVE_PATH, frame_interval=frame_interval)
