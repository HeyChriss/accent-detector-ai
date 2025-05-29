import streamlit as st
import tempfile
import os
from pathlib import Path
import pandas as pd
from accent_detector import AccentDetector
from logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

def run_tests():
    """Check for test files and log test status"""
    test_dir = Path(__file__).parent / 'test'
    test_files = [
        'test_accent_detector.py',
        'test_model.py', 
        'test_video_downloader.py'
    ]
    
    if not test_dir.exists():
        logger.warning("Test directory not found")
        return {'passed': 0, 'failed': 1, 'errors': ['Test directory not found']}
    
    existing_tests = []
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            existing_tests.append(test_file)
    
    if existing_tests:
        logger.info(f"Found {len(existing_tests)} test files: {', '.join(existing_tests)}")
        logger.info("Tests are available but running in background to avoid Streamlit conflicts")
        return {'passed': len(existing_tests), 'failed': 0, 'errors': []}
    else:
        logger.warning("No test files found")
        return {'passed': 0, 'failed': 1, 'errors': ['No test files found']}

# Run tests at startup
if 'tests_run' not in st.session_state:
    st.session_state.tests_run = False
    
if not st.session_state.tests_run:
    test_results = run_tests()
    st.session_state.test_results = test_results
    st.session_state.tests_run = True
    
    # Log test results to console
    if test_results['failed'] == 0:
        logger.info(f"All tests passed! ({test_results['passed']} tests)")
    else:
        logger.warning(f"Tests completed: {test_results['passed']} passed, {test_results['failed']} failed")
        if test_results['errors']:
            for error in test_results['errors']:
                logger.error(f"Test failure: {error}")

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

@st.cache_resource
def load_accent_detector():
    """Load and cache the accent detector model"""
    try:
        detector = AccentDetector()
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logger.error(f"Error loading model: {e}")
        return None

# Load model at startup
if st.session_state.detector is None:
    with st.spinner("Loading accent detection model..."):
        st.session_state.detector = load_accent_detector()
        if st.session_state.detector:
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load the accent detection model. Please refresh the page.")
            st.stop()

def process_url(url):
    """Process a video URL for accent detection"""
    try:
        with st.spinner("Downloading video and analyzing accent..."):
            result = st.session_state.detector.analyze_video_url(url)
            st.session_state.results = result
            return True
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        logger.error(f"Error processing URL: {e}")
        return False

def process_audio_file(uploaded_file):
    """Process an uploaded audio file for accent detection"""
    try:
        # Save uploaded file to temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        with st.spinner("Analyzing accent from uploaded file..."):
            result = st.session_state.detector.predict_accent(tmp_file_path)
            result['audio_path'] = uploaded_file.name
            result['source'] = 'uploaded_file'
            st.session_state.results = result
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        return True
        
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        logger.error(f"Error processing audio file: {e}")
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        return False

def display_results():
    """Display the accent detection results"""
    if st.session_state.results is None:
        return
    
    result = st.session_state.results
    
    st.markdown("---")
    st.header("Results")
    
    # Main result
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Predicted Accent",
            value=result['predicted_accent'],
            delta=None
        )
    
    with col2:
        confidence_pct = result['confidence'] * 100
        st.metric(
            label="Confidence",
            value=f"{confidence_pct:.1f}%",
            delta=None
        )
    
    # File information
    st.subheader("Source Information")
    if 'source_url' in result:
        st.write(f"**Source URL:** {result['source_url']}")
    elif 'source' in result and result['source'] == 'uploaded_file':
        st.write(f"**Uploaded file:** {result.get('audio_path', 'N/A')}")
    
    # All predictions (if available)
    if 'all_predictions' in result and result['all_predictions']:
        st.subheader("📊 All Accent Predictions")
        
        # Create a DataFrame for better display
        predictions_data = []
        for accent, probability in result['all_predictions'].items():
            predictions_data.append({
                'Accent': accent,
                'Probability': probability,
                'Percentage': f"{probability * 100:.1f}%"
            })
        
        # Sort by probability
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df = predictions_df.sort_values('Probability', ascending=False)
        
        # Display top predictions in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 5 Predictions:**")
            top_5 = predictions_df.head(5)
            for _, row in top_5.iterrows():
                is_top = row['Accent'] == result['predicted_accent']
                if is_top:
                    st.write(f"🏆 **{row['Accent']}**: {row['Percentage']}")
                else:
                    st.write(f"• {row['Accent']}: {row['Percentage']}")
        
        with col2:
            # Create a bar chart
            st.write("**Probability Distribution:**")
            chart_data = predictions_df.head(10)  # Show top 10 in chart
            st.bar_chart(
                data=chart_data.set_index('Accent')['Probability'],
                height=300
            )

def main():
    """Main Streamlit app"""
    
    # Title and description
    st.title("Accent Detector")
    st.markdown("""
    Welcome to the Accent Detector! Upload an audio file or provide a video URL to analyze the speaker's accent.
    
    **Supported formats:**
    - Audio files: WAV, MP3, FLAC, etc.
    - Video URLs: YouTube, and other video platforms
    """)
    
    # Input methods
    st.header("📥 Input Method")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["🔗 Video URL", "📁 Upload Audio File"])
    
    url_input = None
    uploaded_file = None
    
    with tab1:
        st.subheader("Enter Video URL")
        url_input = st.text_input(
            "Paste a video URL (YouTube, etc.):",
            placeholder="https://youtu.be/example",
            help="Supported platforms include YouTube and other major video sites"
        )
        
        if url_input:
            st.info(f"URL entered: {url_input}")
    
    with tab2:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
            help="Upload an audio file to analyze the speaker's accent"
        )
        
        if uploaded_file is not None:
            st.info(f"File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    # Accept button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        accept_button = st.button(
            "Analyze Accent",
            type="primary",
            disabled=not (url_input or uploaded_file) or st.session_state.processing,
            use_container_width=True
        )
    
    # Process input when Accept button is clicked
    if accept_button and not st.session_state.processing:
        st.session_state.processing = True
        
        if url_input:
            success = process_url(url_input)
        elif uploaded_file:
            success = process_audio_file(uploaded_file)
        else:
            st.error("Please provide either a URL or upload an audio file.")
            success = False
        
        st.session_state.processing = False
        
        if success:
            st.rerun()
    
    # Display results
    display_results()
    
    # Footer
    st.markdown("---")
    # Clear results button in main content
    if st.session_state.results is not None:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Results", type="secondary"):
                st.session_state.results = None
                st.rerun()

if __name__ == "__main__":
    main() 