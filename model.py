from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


class AccentDetectionModel:
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.model_name = "HamzaSidhu786/speech-accent-detection"
        
    def load_model(self):
        """Load the model and feature extractor from Hugging Face"""
        try:
            logger.info(f"Loading accent detection model: {self.model_name}")
            
            # Load feature extractor (not processor, since this is wav2vec2-based)
            logger.info("Loading feature extractor...")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            
            logger.info("Model and feature extractor loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def is_loaded(self):
        """Check if model and feature extractor are loaded"""
        return self.model is not None and self.feature_extractor is not None
    
    def get_model(self):
        """Get the loaded model"""
        if not self.is_loaded():
            raise Exception("Model not loaded. Call load_model() first.")
        return self.model
    
    def get_feature_extractor(self):
        """Get the loaded feature extractor"""
        if not self.is_loaded():
            raise Exception("Feature extractor not loaded. Call load_model() first.")
        return self.feature_extractor


# Global model instance
accent_model = AccentDetectionModel()


def load_accent_model():
    """Load the accent detection model (convenience function)"""
    if not accent_model.is_loaded():
        accent_model.load_model()
    return accent_model


def get_model_and_feature_extractor():
    """Get the loaded model and feature extractor"""
    if not accent_model.is_loaded():
        accent_model.load_model()
    return accent_model.get_model(), accent_model.get_feature_extractor()


if __name__ == "__main__":
    # Test loading the model
    try:
        print("Testing model loading...")
        model_instance = load_accent_model()
        print("Model loaded successfully!")
        print(f"Model type: {type(model_instance.get_model())}")
        print(f"Feature extractor type: {type(model_instance.get_feature_extractor())}")
    except Exception as e:
        print(f"Error: {e}") 