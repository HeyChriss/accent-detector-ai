import sys
import torch
import librosa
import numpy as np
from pathlib import Path
from model import get_model_and_feature_extractor
from video_downloader import VideoDownloader
from logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


class AccentDetector:
    def __init__(self):
        """Initialize the accent detector"""
        self.model = None
        self.feature_extractor = None
        self.load_model()
    
    def load_model(self):
        """Load the model and feature extractor"""
        try:
            logger.info("Loading accent detection model...")
            self.model, self.feature_extractor = get_model_and_feature_extractor()
            logger.info("Accent detector ready!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_accent(self, audio_path):
        """
        Predict the accent from an audio file
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary containing predicted accent and confidence scores
        """
        try:
            logger.info(f"Analyzing accent from: {audio_path}")
            
            # Check if file exists
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio file
            logger.info("Loading audio file...")
            audio, sample_rate = librosa.load(audio_path, sr=16000)  # Most models expect 16kHz
            
            # Process audio with the feature extractor
            logger.info("Processing audio...")
            inputs = self.feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            
            # Make prediction
            logger.info("Making prediction...")
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get the predicted class
            predicted_class_id = predictions.argmax().item()
            confidence = predictions.max().item()
            
            # Get class labels (if available in model config)
            if hasattr(self.model.config, 'id2label'):
                predicted_accent = self.model.config.id2label[predicted_class_id]
            else:
                predicted_accent = f"Class_{predicted_class_id}"
            
            # Get all class probabilities
            all_predictions = {}
            if hasattr(self.model.config, 'id2label'):
                for i, prob in enumerate(predictions[0]):
                    label = self.model.config.id2label.get(i, f"Class_{i}")
                    all_predictions[label] = float(prob)
            
            result = {
                'predicted_accent': predicted_accent,
                'confidence': float(confidence),
                'all_predictions': all_predictions
            }
            
            logger.info(f"Prediction complete: {predicted_accent} (confidence: {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting accent: {e}")
            raise
    
    def predict_from_audio_data(self, audio_data, sample_rate=16000):
        """
        Predict accent from raw audio data
        
        Args:
            audio_data: Raw audio data (numpy array)
            sample_rate: Sample rate of the audio
            
        Returns:
            dict: Dictionary containing predicted accent and confidence scores
        """
        try:
            logger.info("Analyzing accent from audio data...")
            
            # Resample if necessary
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Process audio with the feature extractor
            inputs = self.feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get results
            predicted_class_id = predictions.argmax().item()
            confidence = predictions.max().item()
            
            if hasattr(self.model.config, 'id2label'):
                predicted_accent = self.model.config.id2label[predicted_class_id]
            else:
                predicted_accent = f"Class_{predicted_class_id}"
            
            result = {
                'predicted_accent': predicted_accent,
                'confidence': float(confidence)
            }
            
            logger.info(f"Prediction complete: {predicted_accent} (confidence: {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting accent from audio data: {e}")
            raise

    def analyze_video_url(self, url):
        """
        Complete pipeline: download video from URL, extract audio, and predict accent
        
        Args:
            url (str): Video URL to analyze
            
        Returns:
            dict: Dictionary containing predicted accent, confidence scores, and file paths
        """
        try:
            logger.info(f"Starting accent analysis for URL: {url}")
            
            # Initialize video downloader
            downloader = VideoDownloader()
            
            # Download video and extract audio
            logger.info("Downloading video and extracting audio...")
            download_result = downloader.process_url(url, extract_audio_flag=True, audio_format='wav')
            
            audio_path = download_result['audio_path']
            video_path = download_result['video_path']
            
            # Predict accent from the extracted audio
            prediction_result = self.predict_accent(audio_path)
            
            # Add file paths to result
            prediction_result['video_path'] = video_path
            prediction_result['audio_path'] = audio_path
            prediction_result['source_url'] = url
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error analyzing video URL: {e}")
            raise

