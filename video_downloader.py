import os
import sys
import tempfile
import validators
import yt_dlp
import requests
from pathlib import Path
from moviepy.editor import VideoFileClip
from urllib.parse import urlparse
from logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


class VideoDownloader:
    def __init__(self, output_dir="downloads"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # yt-dlp options
        self.ydl_opts = {
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'format': 'best',
            'noplaylist': True,
        }
    
    def validate_url(self, url):
        # Handle empty or None URLs
        if not url or not isinstance(url, str) or len(url.strip()) == 0:
            return False
        
        try:
            result = validators.url(url.strip())
            return result is True
        except Exception:
            return False
    
    def is_direct_video_link(self, url):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        return any(path.endswith(ext) for ext in video_extensions)
    
    def download_direct_video(self, url):
        try:
            logger.info(f"Downloading direct video from: {url}")
            
            # Get filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                filename = "downloaded_video.mp4"
            
            filepath = self.output_dir / filename
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Successfully downloaded video to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading direct video: {e}")
            raise
    
    def download_with_ytdlp(self, url):
        try:
            logger.info(f"Downloading video using yt-dlp from: {url}")
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract info first to get the title
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'video')
                
                # Download the video
                ydl.download([url])
                
                # Find the downloaded file
                for file in self.output_dir.glob(f"{video_title}.*"):
                    if file.suffix in ['.mp4', '.webm', '.mkv', '.avi']:
                        logger.info(f"Successfully downloaded video to: {file}")
                        return str(file)
                
                # Fallback: find any recently created video file
                video_files = []
                for ext in ['*.mp4', '*.webm', '*.mkv', '*.avi']:
                    video_files.extend(self.output_dir.glob(ext))
                
                if video_files:
                    # Return the most recently created file
                    latest_file = max(video_files, key=os.path.getctime)
                    logger.info(f"Found downloaded video: {latest_file}")
                    return str(latest_file)
                
                raise Exception("Could not find downloaded video file")

        # Error handling for youtube download errors        
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e).lower()
            if "403" in error_msg or "forbidden" in error_msg:
                logger.error("HTTP 403 Forbidden error detected!")
                print("YOUTUBE ACCESS ERROR (403 Forbidden)")
                print("This error typically occurs due to:")
                print("1. Browser cookies interfering with yt-dlp")
                print("2. Outdated yt-dlp version")
                print("\n SOLUTIONS:")
                print("1. Clear your browser cookies for YouTube")
                print("2. Update yt-dlp: pip install --upgrade yt-dlp")
                print("3. Try again in a few minutes")
                print("4. Use a different video URL for testing")
                raise Exception("YouTube access blocked (403 Forbidden). See solutions above.")
            else:
                logger.error(f"yt-dlp download error: {e}")
                raise
        except Exception as e:
            logger.error(f"Error downloading with yt-dlp: {e}")
            raise
    
    def extract_audio(self, video_path, audio_format='wav'):
        try:
            logger.info(f"Extracting audio from: {video_path}")
            
            video_path = Path(video_path)
            audio_path = video_path.parent / f"{video_path.stem}.{audio_format}"
            
            # Extract audio using moviepy
            with VideoFileClip(str(video_path)) as video:
                audio = video.audio
                audio.write_audiofile(str(audio_path), logger=None)
            
            logger.info(f"Successfully extracted audio to: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def process_url(self, url, extract_audio_flag=True, audio_format='wav'):
        """
        Main method to process a video URL: download video and optionally extract audio
        """
        try:
            # Validate URL
            if not self.validate_url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            logger.info(f"Processing URL: {url}")
            
            # Download video
            if self.is_direct_video_link(url):
                video_path = self.download_direct_video(url)
            else:
                video_path = self.download_with_ytdlp(url)
            
            result = {
                'video_path': video_path,
                'audio_path': None
            }
            
            # Extract audio if requested
            if extract_audio_flag:
                audio_path = self.extract_audio(video_path, audio_format)
                result['audio_path'] = audio_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            raise
