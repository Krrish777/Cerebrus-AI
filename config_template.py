# Configuration for YouTube Audio Transcriber
# Copy this file to config.py and fill in your API key

# AssemblyAI API Configuration
ASSEMBLYAI_API_KEY = "YOUR_API_KEY_HERE"

# Audio Processing Settings
DEFAULT_AUDIO_QUALITY = "best"  # Options: best, high, medium, low
DEFAULT_MAX_DURATION = 600      # Maximum video duration in seconds (10 minutes default)
CLEANUP_AUDIO = True            # Whether to delete temporary audio files
CACHE_AUDIO = True              # Whether to cache downloaded audio files

# Transcription Settings
ENABLE_SPEAKER_DIARIZATION = True
ENABLE_SENTIMENT_ANALYSIS = True
ENABLE_ENTITY_DETECTION = True
ENABLE_AUTO_HIGHLIGHTS = True
ENABLE_AUTO_CHAPTERS = True
ENABLE_SUMMARIZATION = True

# Paths
TEMP_AUDIO_DIR = "temp_audio"
CACHE_AUDIO_DIR = "audio_cache"