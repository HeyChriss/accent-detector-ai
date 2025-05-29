import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import model
from model import AccentDetectionModel, load_accent_model, get_model_and_feature_extractor

# Add the project directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

def test_init_confirmation():

    # Setup

    
    # Exercise
    detector = AccentDetectionModel()
    
    # Verify
    assert detector.model is None
    assert detector.feature_extractor is None
    assert detector.model_name == "HamzaSidhu786/speech-accent-detection"
    
    # Teardown


def test_is_loaded_when_not_loaded_confirmation():
    # Setup
    detector = AccentDetectionModel()
    
    # Exercise
    result = detector.is_loaded()
    
    # Verify
    assert result is False
    
    # Teardown



def test_is_loaded_when_partially_loaded_edge():
    # Setup
    detector = AccentDetectionModel()
    detector.model = Mock()  # Only model loaded, not feature_extractor
    
    # Exercise
    result = detector.is_loaded()
    
    # Verify
    assert result is False
    
    # Teardown


def test_is_loaded_when_fully_loaded_confirmation():
    # Setup
    detector = AccentDetectionModel()
    detector.model = Mock()
    detector.feature_extractor = Mock()
    
    # Exercise
    result = detector.is_loaded()
    
    # Verify
    assert result is True
    
    # Teardown


@patch('model.AutoModelForAudioClassification.from_pretrained')
@patch('model.Wav2Vec2FeatureExtractor.from_pretrained')
@patch('model.logger')
def test_load_model_success_confirmation(mock_logger, mock_feature_extractor, mock_model):
    # Setup
    mock_feature_extractor_instance = Mock()
    mock_model_instance = Mock()
    mock_feature_extractor.return_value = mock_feature_extractor_instance
    mock_model.return_value = mock_model_instance
    
    detector = AccentDetectionModel()
    
    # Exercise
    result = detector.load_model()
    
    # Verify
    assert result is True
    assert detector.feature_extractor == mock_feature_extractor_instance
    assert detector.model == mock_model_instance
    mock_feature_extractor.assert_called_once_with("HamzaSidhu786/speech-accent-detection")
    mock_model.assert_called_once_with("HamzaSidhu786/speech-accent-detection")
    
    # Teardown


@patch('model.Wav2Vec2FeatureExtractor.from_pretrained')
@patch('model.logger')
def test_load_model_feature_extractor_error(mock_logger, mock_feature_extractor):
    # Setup
    mock_feature_extractor.side_effect = Exception("Feature extractor loading failed")
    detector = AccentDetectionModel()
    
    # Exercise & Verify
    try:
        detector.load_model()
        assert False, "Expected exception was not raised"
    except Exception as e:
        assert str(e) == "Feature extractor loading failed"
        mock_logger.error.assert_called_once()
    
    # Teardown


@patch('model.AutoModelForAudioClassification.from_pretrained')
@patch('model.Wav2Vec2FeatureExtractor.from_pretrained')
@patch('model.logger')
def test_load_model_model_error(mock_logger, mock_feature_extractor, mock_model):
    # Setup
    mock_feature_extractor.return_value = Mock()
    mock_model.side_effect = Exception("Model loading failed")
    detector = AccentDetectionModel()
    
    # Exercise & Verify
    try:
        detector.load_model()
        assert False, "Expected exception was not raised"
    except Exception as e:
        assert str(e) == "Model loading failed"
        mock_logger.error.assert_called_once()
    
    # Teardown


def test_get_model_when_not_loaded_error():
    # Setup
    detector = AccentDetectionModel()
    
    # Exercise & Verify
    try:
        detector.get_model()
        assert False, "Expected exception was not raised"
    except Exception as e:
        assert str(e) == "Model not loaded. Call load_model() first."
    
    # Teardown


def test_get_model_when_loaded_confirmation():
    # Setup
    detector = AccentDetectionModel()
    mock_model = Mock()
    detector.model = mock_model
    detector.feature_extractor = Mock()  # Both need to be loaded
    
    # Exercise
    result = detector.get_model()
    
    # Verify
    assert result == mock_model
    
    # Teardown


def test_get_feature_extractor_when_not_loaded_error():
    # Setup
    detector = AccentDetectionModel()
    
    # Exercise & Verify
    try:
        detector.get_feature_extractor()
        assert False, "Expected exception was not raised"
    except Exception as e:
        assert str(e) == "Feature extractor not loaded. Call load_model() first."
    
    # Teardown


def test_get_feature_extractor_when_loaded_confirmation():
    # Setup
    detector = AccentDetectionModel()
    mock_feature_extractor = Mock()
    detector.model = Mock()  # Both need to be loaded
    detector.feature_extractor = mock_feature_extractor
    
    # Exercise
    result = detector.get_feature_extractor()
    
    # Verify
    assert result == mock_feature_extractor
    
    # Teardown


@patch.object(AccentDetectionModel, 'load_model')
def test_load_accent_model_when_not_loaded_confirmation(mock_load_model):
    # Setup
    original_accent_model = model.accent_model
    model.accent_model = AccentDetectionModel()
    mock_load_model.return_value = True
    
    # Exercise
    result = load_accent_model()
    
    # Verify
    assert result == model.accent_model
    mock_load_model.assert_called_once()
    
    # Teardown
    model.accent_model = original_accent_model


@patch.object(AccentDetectionModel, 'load_model')
@patch.object(AccentDetectionModel, 'is_loaded')
def test_load_accent_model_when_already_loaded_edge(mock_is_loaded, mock_load_model):
    # Setup
    original_accent_model = model.accent_model
    model.accent_model = AccentDetectionModel()
    mock_is_loaded.return_value = True
    
    # Exercise
    result = load_accent_model()
    
    # Verify
    assert result == model.accent_model
    mock_load_model.assert_not_called()
    
    # Teardown
    model.accent_model = original_accent_model


@patch.object(AccentDetectionModel, 'load_model')
def test_load_accent_model_propagates_exception_error(mock_load_model):
    # Setup
    original_accent_model = model.accent_model
    model.accent_model = AccentDetectionModel()
    mock_load_model.side_effect = Exception("Loading failed")
    
    # Exercise & Verify
    try:
        load_accent_model()
        assert False, "Expected exception was not raised"
    except Exception as e:
        assert str(e) == "Loading failed"
    
    # Teardown
    model.accent_model = original_accent_model


@patch.object(AccentDetectionModel, 'get_feature_extractor')
@patch.object(AccentDetectionModel, 'get_model')
@patch.object(AccentDetectionModel, 'load_model')
@patch.object(AccentDetectionModel, 'is_loaded')
def test_get_model_and_feature_extractor_confirmation(mock_is_loaded, mock_load_model, 
                                                     mock_get_model, mock_get_feature_extractor):
    # Setup
    original_accent_model = model.accent_model
    model.accent_model = AccentDetectionModel()
    mock_is_loaded.return_value = True
    mock_model = Mock()
    mock_feature_extractor = Mock()
    mock_get_model.return_value = mock_model
    mock_get_feature_extractor.return_value = mock_feature_extractor
    
    # Exercise
    model_result, feature_extractor_result = get_model_and_feature_extractor()
    
    # Verify
    assert model_result == mock_model
    assert feature_extractor_result == mock_feature_extractor
    mock_load_model.assert_not_called()  # Should not load since already loaded
    
    # Teardown
    model.accent_model = original_accent_model


@patch.object(AccentDetectionModel, 'get_feature_extractor')
@patch.object(AccentDetectionModel, 'get_model')
@patch.object(AccentDetectionModel, 'load_model')
@patch.object(AccentDetectionModel, 'is_loaded')
def test_get_model_and_feature_extractor_loads_when_not_loaded_edge(mock_is_loaded, mock_load_model,
                                                                   mock_get_model, mock_get_feature_extractor):
    # Setup
    original_accent_model = model.accent_model
    model.accent_model = AccentDetectionModel()
    mock_is_loaded.return_value = False
    mock_load_model.return_value = True
    mock_model = Mock()
    mock_feature_extractor = Mock()
    mock_get_model.return_value = mock_model
    mock_get_feature_extractor.return_value = mock_feature_extractor
    
    # Exercise
    model_result, feature_extractor_result = get_model_and_feature_extractor()
    
    # Verify
    assert model_result == mock_model
    assert feature_extractor_result == mock_feature_extractor
    mock_load_model.assert_called_once()
    
    # Teardown
    model.accent_model = original_accent_model


@patch.object(AccentDetectionModel, 'get_model')
@patch.object(AccentDetectionModel, 'load_model')
@patch.object(AccentDetectionModel, 'is_loaded')
def test_get_model_and_feature_extractor_propagates_get_model_error(mock_is_loaded, 
                                                                   mock_load_model, mock_get_model):
    # Setup
    original_accent_model = model.accent_model
    model.accent_model = AccentDetectionModel()
    mock_is_loaded.return_value = True
    mock_get_model.side_effect = Exception("Get model failed")
    
    # Exercise & Verify
    try:
        get_model_and_feature_extractor()
        assert False, "Expected exception was not raised"
    except Exception as e:
        assert str(e) == "Get model failed"
    
    # Teardown
    model.accent_model = original_accent_model


@patch.object(AccentDetectionModel, 'get_feature_extractor')
@patch.object(AccentDetectionModel, 'get_model')
@patch.object(AccentDetectionModel, 'load_model')
@patch.object(AccentDetectionModel, 'is_loaded')
def test_get_model_and_feature_extractor_propagates_get_feature_extractor_error(mock_is_loaded, 
                                                                               mock_load_model, mock_get_model,
                                                                               mock_get_feature_extractor):
    # Setup
    original_accent_model = model.accent_model
    model.accent_model = AccentDetectionModel()
    mock_is_loaded.return_value = True
    mock_get_model.return_value = Mock()
    mock_get_feature_extractor.side_effect = Exception("Get feature extractor failed")
    
    # Exercise & Verify
    try:
        get_model_and_feature_extractor()
        assert False, "Expected exception was not raised"
    except Exception as e:
        assert str(e) == "Get feature extractor failed"
    
    # Teardown
    model.accent_model = original_accent_model


def test_model_name_is_correct_confirmation():

    # Exercise
    detector = AccentDetectionModel()
    
    # Verify
    assert detector.model_name == "HamzaSidhu786/speech-accent-detection"
    
    # Teardown


def test_model_name_is_string_edge():
    # Setup
    
    # Exercise
    detector = AccentDetectionModel()
    
    # Verify
    assert isinstance(detector.model_name, str)
    assert len(detector.model_name) > 0
    
    # Teardown


def run_all_tests():
    """Run all test functions"""
    test_functions = [
        test_init_confirmation,
        test_is_loaded_when_not_loaded_confirmation,
        test_is_loaded_when_partially_loaded_edge,
        test_is_loaded_when_fully_loaded_confirmation,
        test_load_model_success_confirmation,
        test_load_model_feature_extractor_error,
        test_load_model_model_error,
        test_get_model_when_not_loaded_error,
        test_get_model_when_loaded_confirmation,
        test_get_feature_extractor_when_not_loaded_error,
        test_get_feature_extractor_when_loaded_confirmation,
        test_load_accent_model_when_not_loaded_confirmation,
        test_load_accent_model_when_already_loaded_edge,
        test_load_accent_model_propagates_exception_error,
        test_get_model_and_feature_extractor_confirmation,
        test_get_model_and_feature_extractor_loads_when_not_loaded_edge,
        test_get_model_and_feature_extractor_propagates_get_model_error,
        test_get_model_and_feature_extractor_propagates_get_feature_extractor_error,
        test_model_name_is_correct_confirmation,
        test_model_name_is_string_edge
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            print(f"✓ PASSED: {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_func.__name__} - {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    # Run all tests
    success = run_all_tests() 