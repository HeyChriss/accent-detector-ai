# 🎤 Accent Detector AI

A powerful AI-powered accent detection application built with Streamlit that can analyze speech accents from audio files or video URLs. The app uses advanced machine learning to identify different English accents with high accuracy.

## 🌟 Features

- **Audio File Analysis**: Upload audio files (WAV, MP3, FLAC, M4A, OGG) for accent detection
- **Video URL Analysis**: Extract audio from video URLs (YouTube and other platforms) and analyze accents
- **Real-time Results**: Get instant accent predictions with confidence scores
- **Multiple Accent Support**: Detects various English accents including American, British, Australian, and more
- **Detailed Analytics**: View probability distributions for all detected accents
- **Clean Interface**: Modern, user-friendly Streamlit interface

## 🤖 AI Model

This application uses the **HamzaSidhu786/speech-accent-detection** model from Hugging Face:

🔗 **Model Link**: [https://huggingface.co/HamzaSidhu786/speech-accent-detection](https://huggingface.co/HamzaSidhu786/speech-accent-detection)

### Model Details:
- **Base Architecture**: Fine-tuned Facebook Wav2Vec2-base model
- **Training Dataset**: CSTR-Edinburgh/VCTK dataset
- **Accuracy**: 99.55% on evaluation set
- **Framework**: PyTorch + Transformers
- **Model Size**: 94.6M parameters
- **License**: Apache 2.0

The model achieves exceptional performance with a validation loss of 0.0441 and was trained for 10 epochs using advanced hyperparameters and mixed precision training.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/heychriss/accent-detector-ai.git
   cd accent-detector-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 How to Use

### Method 1: Upload Audio File
1. Click on the "📁 Upload Audio File" tab
2. Choose an audio file from your device
3. Click "Analyze Accent" button
4. View the results with confidence scores

### Method 2: Video URL Analysis
1. Click on the "🔗 Video URL" tab
2. Paste a video URL (YouTube, etc.)
3. Click "Analyze Accent" button
4. The app will download, extract audio, and analyze the accent

### Understanding Results
- **Predicted Accent**: The most likely accent detected
- **Confidence**: Percentage confidence in the prediction
- **Probability Distribution**: Shows all possible accents with their probabilities
- **Source Information**: Details about the analyzed file or URL

## 🏗️ Project Structure

```
accent-detector-ai/
├── streamlit_app.py          # Main Streamlit application
├── accent_detector.py        # Core accent detection logic
├── model.py                 # Model loading and management
├── video_downloader.py      # Video URL processing
├── logger.py               # Logging configuration
├── requirements.txt        # Python dependencies
├── test/                   # Test files
│   ├── test_accent_detector.py
│   ├── test_model.py
│   └── test_video_downloader.py
├── .streamlit/            # Streamlit configuration
│   └── config.toml
└── README.md             # This file
```

## 🧪 Testing

The application includes comprehensive tests that run automatically on startup:

- **Test Coverage**: Accent detector, model loading, video downloader
- **Automated Testing**: Tests run silently in background during app initialization
- **Console Logging**: Test results are logged to console for development monitoring

## 🔧 Technical Details

### Core Technologies
- **Frontend**: Streamlit for web interface
- **ML Framework**: PyTorch + Transformers
- **Audio Processing**: librosa for audio analysis
- **Video Processing**: yt-dlp for video URL handling
- **Logging**: Python logging with custom configuration

### Key Components
- **AccentDetector**: Main class handling accent prediction
- **ModelManager**: Handles model loading and caching
- **VideoDownloader**: Manages video URL processing and audio extraction
- **Logger**: Centralized logging system

### Performance Optimizations
- **Model Caching**: Models are cached using Streamlit's `@st.cache_resource`
- **Efficient Processing**: Optimized audio preprocessing and inference
- **Memory Management**: Proper cleanup of temporary files

## 📊 Supported Accents

The model can detect various English accents including but not limited to:
- American English
- British English
- Australian English
- Canadian English
- Irish English
- Scottish English
- And more regional variations

*For complete accent list and detailed model performance, visit the [model page](https://huggingface.co/HamzaSidhu786/speech-accent-detection).*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Model Creator**: [HamzaSidhu786](https://huggingface.co/HamzaSidhu786) for the excellent accent detection model
- **Base Model**: Facebook's Wav2Vec2 team for the foundational architecture
- **Dataset**: CSTR-Edinburgh for the VCTK dataset
- **Community**: Hugging Face community for model hosting and tools

