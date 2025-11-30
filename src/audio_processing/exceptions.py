"""Custom exceptions for audio processing module."""


class AudioProcessingError(Exception):
    """Base exception for all audio processing errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize audio processing error.
        
        :param message: Error message
        :param details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AudioProcessingError):
    """Raised when configuration is invalid or missing."""
    pass


class TranscriptionError(AudioProcessingError):
    """Raised when transcription fails."""
    pass


class ProviderError(AudioProcessingError):
    """Raised when a transcription provider encounters an error."""
    pass


class ChunkingError(AudioProcessingError):
    """Raised when audio chunking fails."""
    pass


class ExtractionError(AudioProcessingError):
    """Raised when data extraction fails."""
    pass


class ValidationError(AudioProcessingError):
    """Raised when input validation fails."""
    pass


class DocumentBuildError(AudioProcessingError):
    """Raised when document building fails."""
    pass
