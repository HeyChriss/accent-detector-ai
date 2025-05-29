import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import os
import sys
import numpy as np
import torch

# Add parent directory to path to import from root
sys.path.insert(0, str(Path(__file__).parent.parent))
from accent_detector import AccentDetector


def test_accent_detector_init_success():
    """Test AccentDetector initialization with successful model loading"""
    # Setup
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_model = Mock()
        mock_feature_extractor = Mock()
        mock_get_model.return_value = (mock_model, mock_feature_extractor)
        
        # Exercise
        detector = AccentDetector()
        
        # Verify
        assert detector.model == mock_model
        assert detector.feature_extractor == mock_feature_extractor
        mock_get_model.assert_called_once()
        
        # Teardown
        # (No teardown needed)


def test_accent_detector_init_failure():
    """Test AccentDetector initialization with model loading failure"""
    # Setup
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_get_model.side_effect = Exception("Model loading failed")
        
        # Exercise & Verify
        try:
            AccentDetector()
            assert False, "Expected exception was not raised"
        except Exception as e:
            assert "Model loading failed" in str(e)
        
        # Teardown
        # (No teardown needed)


@patch('accent_detector.librosa.load')
@patch('accent_detector.torch.no_grad')
def test_predict_accent_success(mock_no_grad, mock_librosa_load):
    """Test successful accent prediction from audio file"""
    # Setup
    temp_dir = tempfile.mkdtemp()
    test_audio_path = os.path.join(temp_dir, "test_audio.wav")
    Path(test_audio_path).touch()  # Create fake audio file
    
    # Mock audio loading
    mock_audio_data = np.array([0.1, 0.2, 0.3])
    mock_librosa_load.return_value = (mock_audio_data, 16000)
    
    # Mock model and feature extractor
    mock_model = Mock()
    mock_feature_extractor = Mock()
    mock_inputs = {"input_values": torch.tensor([[0.1, 0.2, 0.3]])}
    mock_feature_extractor.return_value = mock_inputs
    
    # Mock model output
    mock_logits = torch.tensor([[0.2, 0.8, 0.1]])  # 3 classes
    mock_outputs = Mock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    
    # Mock model config
    mock_model.config.id2label = {0: "american", 1: "british", 2: "australian"}
    
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_get_model.return_value = (mock_model, mock_feature_extractor)
        detector = AccentDetector()
        
        try:
            # Exercise
            result = detector.predict_accent(test_audio_path)
            
            # Verify
            assert 'predicted_accent' in result
            assert 'confidence' in result
            assert 'all_predictions' in result
            assert result['predicted_accent'] == "british"  # Highest probability class
            assert isinstance(result['confidence'], float)
            assert isinstance(result['all_predictions'], dict)
            mock_librosa_load.assert_called_once_with(test_audio_path, sr=16000)
            mock_feature_extractor.assert_called_once()
            
        finally:
            # Teardown
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_predict_accent_file_not_found():
    """Test accent prediction with non-existent audio file"""
    # Setup
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_get_model.return_value = (Mock(), Mock())
        detector = AccentDetector()
        non_existent_path = "/fake/path/audio.wav"
        
        # Exercise & Verify
        try:
            detector.predict_accent(non_existent_path)
            assert False, "Expected FileNotFoundError was not raised"
        except FileNotFoundError as e:
            assert "Audio file not found" in str(e)
        
        # Teardown
        # (No teardown needed)


@patch('accent_detector.librosa.load')
def test_predict_accent_librosa_error(mock_librosa_load):
    """Test accent prediction with librosa loading error"""
    # Setup
    temp_dir = tempfile.mkdtemp()
    test_audio_path = os.path.join(temp_dir, "test_audio.wav")
    Path(test_audio_path).touch()
    
    mock_librosa_load.side_effect = Exception("Audio loading failed")
    
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_get_model.return_value = (Mock(), Mock())
        detector = AccentDetector()
        
        try:
            # Exercise & Verify
            try:
                detector.predict_accent(test_audio_path)
                assert False, "Expected exception was not raised"
            except Exception as e:
                assert "Audio loading failed" in str(e)
                
        finally:
            # Teardown
            shutil.rmtree(temp_dir, ignore_errors=True)


@patch('accent_detector.librosa.resample')
@patch('accent_detector.torch.no_grad')
def test_predict_from_audio_data_resample(mock_no_grad, mock_resample):
    """Test accent prediction from audio data with resampling"""
    # Setup
    mock_audio_data = np.array([0.1, 0.2, 0.3])
    mock_resampled_data = np.array([0.15, 0.25, 0.35])
    mock_resample.return_value = mock_resampled_data
    
    # Mock model and feature extractor
    mock_model = Mock()
    mock_feature_extractor = Mock()
    mock_inputs = {"input_values": torch.tensor([[0.15, 0.25, 0.35]])}
    mock_feature_extractor.return_value = mock_inputs
    
    # Mock model output
    mock_logits = torch.tensor([[0.3, 0.7]])
    mock_outputs = Mock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    mock_model.config.id2label = {0: "accent1", 1: "accent2"}
    
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_get_model.return_value = (mock_model, mock_feature_extractor)
        detector = AccentDetector()
        
        # Exercise
        result = detector.predict_from_audio_data(mock_audio_data, sample_rate=22050)
        
        # Verify
        assert 'predicted_accent' in result
        assert 'confidence' in result
        assert result['predicted_accent'] == "accent2"
        mock_resample.assert_called_once_with(mock_audio_data, orig_sr=22050, target_sr=16000)
        
        # Teardown
        # (No teardown needed)


@patch('accent_detector.torch.no_grad')
def test_predict_from_audio_data_no_resample(mock_no_grad):
    """Test accent prediction from audio data without resampling"""
    # Setup
    mock_audio_data = np.array([0.1, 0.2, 0.3])
    
    # Mock model and feature extractor
    mock_model = Mock()
    mock_feature_extractor = Mock()
    mock_inputs = {"input_values": torch.tensor([[0.1, 0.2, 0.3]])}
    mock_feature_extractor.return_value = mock_inputs
    
    # Mock model output
    mock_logits = torch.tensor([[0.6, 0.4]])
    mock_outputs = Mock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    mock_model.config.id2label = {0: "accent1", 1: "accent2"}
    
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_get_model.return_value = (mock_model, mock_feature_extractor)
        detector = AccentDetector()
        
        # Exercise
        result = detector.predict_from_audio_data(mock_audio_data, sample_rate=16000)
        
        # Verify
        assert result['predicted_accent'] == "accent1"
        mock_feature_extractor.assert_called_with(mock_audio_data, sampling_rate=16000, 
                                                return_tensors="pt", padding=True)
        
        # Teardown
        # (No teardown needed)




@patch.object(AccentDetector, 'predict_accent')
@patch('accent_detector.VideoDownloader')
def test_analyze_video_url_success(mock_video_downloader_class, mock_predict):
    """Test successful video URL analysis"""
    # Setup
    test_url = "https://youtube.com/watch?v=test"
    test_video_path = "/fake/video.mp4"
    test_audio_path = "/fake/audio.wav"
    
    # Mock VideoDownloader
    mock_downloader = Mock()
    mock_download_result = {
        'video_path': test_video_path,
        'audio_path': test_audio_path
    }
    mock_downloader.process_url.return_value = mock_download_result
    mock_video_downloader_class.return_value = mock_downloader
    
    # Mock prediction result
    mock_prediction_result = {
        'predicted_accent': 'american',
        'confidence': 0.85,
        'all_predictions': {'american': 0.85, 'british': 0.15}
    }
    mock_predict.return_value = mock_prediction_result
    
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_get_model.return_value = (Mock(), Mock())
        detector = AccentDetector()
        
        # Exercise
        result = detector.analyze_video_url(test_url)
        
        # Verify
        assert result['predicted_accent'] == 'american'
        assert result['confidence'] == 0.85
        assert result['video_path'] == test_video_path
        assert result['audio_path'] == test_audio_path
        assert result['source_url'] == test_url
        mock_downloader.process_url.assert_called_once_with(test_url, extract_audio_flag=True, audio_format='wav')
        mock_predict.assert_called_once_with(test_audio_path)
        
        # Teardown
        # (No teardown needed)


@patch('accent_detector.VideoDownloader')
def test_analyze_video_url_download_failure(mock_video_downloader_class):
    """Test video URL analysis with download failure"""
    # Setup
    test_url = "https://youtube.com/watch?v=test"
    
    # Mock VideoDownloader to raise exception
    mock_downloader = Mock()
    mock_downloader.process_url.side_effect = Exception("Download failed")
    mock_video_downloader_class.return_value = mock_downloader
    
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_get_model.return_value = (Mock(), Mock())
        detector = AccentDetector()
        
        # Exercise & Verify
        try:
            detector.analyze_video_url(test_url)
            assert False, "Expected exception was not raised"
        except Exception as e:
            assert "Download failed" in str(e)
        
        # Teardown
        # (No teardown needed)


@patch.object(AccentDetector, 'predict_accent')
@patch('accent_detector.VideoDownloader')
def test_analyze_video_url_prediction_failure(mock_video_downloader_class, mock_predict):
    """Test video URL analysis with prediction failure"""
    # Setup
    test_url = "https://youtube.com/watch?v=test"
    
    # Mock successful download
    mock_downloader = Mock()
    mock_download_result = {
        'video_path': "/fake/video.mp4",
        'audio_path': "/fake/audio.wav"
    }
    mock_downloader.process_url.return_value = mock_download_result
    mock_video_downloader_class.return_value = mock_downloader
    
    # Mock prediction failure
    mock_predict.side_effect = Exception("Prediction failed")
    
    with patch('accent_detector.get_model_and_feature_extractor') as mock_get_model:
        mock_get_model.return_value = (Mock(), Mock())
        detector = AccentDetector()
        
        # Exercise & Verify
        try:
            detector.analyze_video_url(test_url)
            assert False, "Expected exception was not raised"
        except Exception as e:
            assert "Prediction failed" in str(e)
        
        # Teardown
        # (No teardown needed)


def test_main_function_success():
    """Test main function execution with successful analysis"""
    # Setup
    with patch('accent_detector.AccentDetector') as mock_detector_class:
        mock_detector = Mock()
        mock_result = {
            'source_url': 'https://youtu.be/yTiooVPtA3s',
            'video_path': '/fake/video.mp4',
            'audio_path': '/fake/audio.wav',
            'predicted_accent': 'american',
            'confidence': 0.92,
            'all_predictions': {
                'american': 0.92,
                'british': 0.05,
                'australian': 0.03
            }
        }
        mock_detector.analyze_video_url.return_value = mock_result
        mock_detector_class.return_value = mock_detector
        
        with patch('builtins.print') as mock_print:
            with patch('accent_detector.sys.exit') as mock_exit:
                # Import and call main
                from accent_detector import main
                
                # Exercise
                main()
                
                # Verify
                mock_detector_class.assert_called_once()
                mock_detector.analyze_video_url.assert_called_once()
                mock_exit.assert_not_called()  # Should not exit on success
                
        # Teardown
        # (No teardown needed)


def test_main_function_failure():
    """Test main function execution with analysis failure"""
    # Setup
    with patch('accent_detector.AccentDetector') as mock_detector_class:
        mock_detector_class.side_effect = Exception("Initialization failed")
        
        with patch('builtins.print') as mock_print:
            with patch('accent_detector.sys.exit') as mock_exit:
                # Import and call main
                from accent_detector import main
                
                # Exercise
                main()
                
                # Verify
                mock_exit.assert_called_once_with(1)  # Should exit with error code
                
        # Teardown
        # (No teardown needed)


if __name__ == '__main__':
    # Run all test functions
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