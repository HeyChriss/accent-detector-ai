import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from video_downloader import VideoDownloader


def test_video_downloader_init_default():
    # Setup
    temp_dir = None
    downloader = None
    
    try:
        # Exercise
        downloader = VideoDownloader()
        
        # Verify
        assert str(downloader.output_dir) == "downloads"
        assert downloader.output_dir.exists()
        assert isinstance(downloader.ydl_opts, dict)
        assert 'outtmpl' in downloader.ydl_opts
        assert 'format' in downloader.ydl_opts
        assert 'noplaylist' in downloader.ydl_opts
        
    finally:
        # Teardown
        if Path("downloads").exists():
            shutil.rmtree("downloads", ignore_errors=True)


def test_video_downloader_init_custom():
    # Setup
    temp_dir = tempfile.mkdtemp()
    test_output_dir = os.path.join(temp_dir, "test_downloads")
    downloader = None
    
    try:
        # Exercise
        downloader = VideoDownloader(test_output_dir)
        
        # Verify
        assert str(downloader.output_dir) == test_output_dir
        assert downloader.output_dir.exists()
        
    finally:
        # Teardown
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_valid_http_url():
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    valid_url = "http://example.com/video.mp4"
    
    try:
        # Exercise
        result = downloader.validate_url(valid_url)
        
        # Verify
        assert result == True
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_valid_https_url():
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    valid_url = "https://www.youtube.com/watch?v=test"
    
    try:
        # Exercise
        result = downloader.validate_url(valid_url)
        
        # Verify
        assert result == True
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_invalid_url():
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    invalid_url = "not-a-url"
    
    try:
        # Exercise
        result = downloader.validate_url(invalid_url)
        
        # Verify
        assert result == False
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_empty_url():
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    empty_url = ""
    
    try:
        # Exercise
        result = downloader.validate_url(empty_url)
        
        # Verify
        assert result == False
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_is_direct_video_link_mp4():
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    mp4_url = "https://example.com/video.mp4"
    
    try:
        # Exercise
        result = downloader.is_direct_video_link(mp4_url)
        
        # Verify
        assert result == True
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_is_direct_video_link_avi():
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    avi_url = "https://example.com/video.avi"
    
    try:
        # Exercise
        result = downloader.is_direct_video_link(avi_url)
        
        # Verify
        assert result == True
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_is_not_direct_video_link_youtube():
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    youtube_url = "https://www.youtube.com/watch?v=test"
    
    try:
        # Exercise
        result = downloader.is_direct_video_link(youtube_url)
        
        # Verify
        assert result == False
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_is_not_direct_video_link_html():
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    html_url = "https://example.com/page.html"
    
    try:
        # Exercise
        result = downloader.is_direct_video_link(html_url)
        
        # Verify
        assert result == False
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


@patch('video_downloader.requests.get')
def test_download_direct_video_success(mock_get):
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    test_url = "https://example.com/test_video.mp4"
    test_content = b"fake_video_content"
    
    mock_response = Mock()
    mock_response.iter_content.return_value = [test_content]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    try:
        # Exercise
        result_path = downloader.download_direct_video(test_url)
        
        # Verify
        assert os.path.exists(result_path)
        assert result_path.endswith("test_video.mp4")
        with open(result_path, 'rb') as f:
            assert f.read() == test_content
        mock_get.assert_called_once_with(test_url, stream=True)
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


@patch('video_downloader.requests.get')
def test_download_direct_video_no_filename(mock_get):
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    url_no_filename = "https://example.com/"
    test_content = b"fake_video_content"
    
    mock_response = Mock()
    mock_response.iter_content.return_value = [test_content]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    try:
        # Exercise
        result_path = downloader.download_direct_video(url_no_filename)
        
        # Verify
        assert os.path.exists(result_path)
        assert result_path.endswith("downloaded_video.mp4")
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


@patch('video_downloader.requests.get')
def test_download_direct_video_request_error(mock_get):
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    test_url = "https://example.com/test_video.mp4"
    mock_get.side_effect = Exception("Network error")
    
    try:
        # Exercise & Verify
        try:
            downloader.download_direct_video(test_url)
            assert False, "Expected exception was not raised"
        except Exception:
            assert True  # Exception was expected
            
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


@patch('video_downloader.yt_dlp.YoutubeDL')
def test_download_with_ytdlp_success(mock_ytdl_class):
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    test_url = "https://www.youtube.com/watch?v=test"
    test_title = "Test Video"
    test_video_path = os.path.join(temp_dir, f"{test_title}.mp4")
    
    mock_ytdl = Mock()
    mock_ytdl.extract_info.return_value = {'title': test_title}
    mock_ytdl.download.return_value = None
    mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
    
    try:
        # Create fake video file
        Path(test_video_path).touch()
        
        # Exercise
        result_path = downloader.download_with_ytdlp(test_url)
        
        # Verify
        assert result_path == test_video_path
        mock_ytdl.extract_info.assert_called_with(test_url, download=False)
        mock_ytdl.download.assert_called_with([test_url])
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)

@patch('video_downloader.VideoFileClip')
def test_extract_audio_success(mock_video_clip):
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    test_video_path = os.path.join(temp_dir, "test_video.mp4")
    expected_audio_path = os.path.join(temp_dir, "test_video.wav")
    
    # Create fake video file
    Path(test_video_path).touch()
    
    mock_audio = Mock()
    mock_video = Mock()
    mock_video.audio = mock_audio
    mock_video_clip.return_value.__enter__.return_value = mock_video
    
    try:
        # Exercise
        result_path = downloader.extract_audio(test_video_path)
        
        # Verify
        assert result_path == expected_audio_path
        mock_video_clip.assert_called_once_with(test_video_path)
        mock_audio.write_audiofile.assert_called_once_with(expected_audio_path, logger=None)
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


@patch('video_downloader.VideoFileClip')
def test_extract_audio_mp3_format(mock_video_clip):
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    test_video_path = os.path.join(temp_dir, "test_video.mp4")
    expected_mp3_path = os.path.join(temp_dir, "test_video.mp3")
    
    # Create fake video file
    Path(test_video_path).touch()
    
    mock_audio = Mock()
    mock_video = Mock()
    mock_video.audio = mock_audio
    mock_video_clip.return_value.__enter__.return_value = mock_video
    
    try:
        # Exercise
        result_path = downloader.extract_audio(test_video_path, 'mp3')
        
        # Verify
        assert result_path == expected_mp3_path
        mock_audio.write_audiofile.assert_called_once_with(expected_mp3_path, logger=None)
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


@patch.object(VideoDownloader, 'extract_audio')
@patch.object(VideoDownloader, 'download_direct_video')
@patch.object(VideoDownloader, 'is_direct_video_link')
@patch.object(VideoDownloader, 'validate_url')
def test_process_url_direct_video_with_audio(mock_validate, mock_is_direct, mock_download, mock_extract):
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    test_url = "https://example.com/video.mp4"
    test_video_path = os.path.join(temp_dir, "video.mp4")
    test_audio_path = os.path.join(temp_dir, "video.wav")
    
    mock_validate.return_value = True
    mock_is_direct.return_value = True
    mock_download.return_value = test_video_path
    mock_extract.return_value = test_audio_path
    
    try:
        # Exercise
        result = downloader.process_url(test_url, extract_audio_flag=True)
        
        # Verify
        assert result['video_path'] == test_video_path
        assert result['audio_path'] == test_audio_path
        mock_validate.assert_called_once_with(test_url)
        mock_is_direct.assert_called_once_with(test_url)
        mock_download.assert_called_once_with(test_url)
        mock_extract.assert_called_once_with(test_video_path, 'wav')
        
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


@patch.object(VideoDownloader, 'validate_url')
def test_process_url_invalid_url(mock_validate):
    # Setup
    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(temp_dir)
    test_url = "https://example.com/video.mp4"
    mock_validate.return_value = False
    
    try:
        # Exercise & Verify
        try:
            downloader.process_url(test_url)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert "Invalid URL" in str(e)
            
    finally:
        # Teardown
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    # Run all test functions
    import sys
    import inspect
    
    # Get all test functions from current module
    current_module = sys.modules[__name__]
    test_functions = [obj for name, obj in inspect.getmembers(current_module) 
                     if inspect.isfunction(obj) and name.startswith('test_')]
    
    print(f"Running {len(test_functions)} tests...")
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}... ", end="")
            test_func()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed") 