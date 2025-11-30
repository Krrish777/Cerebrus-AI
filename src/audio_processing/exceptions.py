"""
Custom exceptions for audio processing module.

Provides specific exception types for different failure scenarios
in audio transcription and processing operations.
"""


class AudioProcessingException(Exception):
    """Base exception for all audio processing errors."""

    def __init__(self, message: str, original_exception: BaseException | None = None):
        """
        Initialize the audio processing exception.

        :param message: Error message describing the failure
        :param original_exception: Original exception that caused this error
        """
        self.message = message
        self.original_exception = original_exception
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the exception message with optional original exception info."""
        if self.original_exception:
            return f"{self.message} | Caused by: {self.original_exception}"
        return self.message


class TranscriptionServiceUnavailableError(AudioProcessingException):
    """Raised when the transcription service is not available or not installed."""

    def __init__(
        self, 
        service_name: str, 
        original_exception: BaseException | None = None
    ):
        """
        Initialize the transcription service unavailable error.

        :param service_name: Name of the unavailable service
        :param original_exception: Original exception that caused this error
        """
        self.service_name = service_name
        message = f"Transcription service '{service_name}' is not available"
        super().__init__(message, original_exception)


class TranscriptionConfigurationError(AudioProcessingException):
    """Raised when transcription configuration is invalid or incomplete."""

    def __init__(
        self, 
        config_field: str, 
        reason: str, 
        original_exception: BaseException | None = None
    ):
        """
        Initialize the transcription configuration error.

        :param config_field: Configuration field that caused the error
        :param reason: Reason for the configuration error
        :param original_exception: Original exception that caused this error
        """
        self.config_field = config_field
        self.reason = reason
        message = f"Invalid configuration for '{config_field}': {reason}"
        super().__init__(message, original_exception)


class TranscriptionAPIError(AudioProcessingException):
    """Raised when API key is missing or invalid."""

    def __init__(
        self, 
        api_name: str, 
        reason: str, 
        original_exception: BaseException | None = None
    ):
        """
        Initialize the transcription API error.

        :param api_name: Name of the API that failed
        :param reason: Reason for the API error
        :param original_exception: Original exception that caused this error
        """
        self.api_name = api_name
        self.reason = reason
        message = f"API error for '{api_name}': {reason}"
        super().__init__(message, original_exception)


class AudioSourceError(AudioProcessingException):
    """Raised when audio source is invalid, inaccessible, or unsupported."""

    def __init__(
        self, 
        source: str, 
        reason: str, 
        original_exception: BaseException | None = None
    ):
        """
        Initialize the audio source error.

        :param source: Audio source that caused the error
        :param reason: Reason for the source error
        :param original_exception: Original exception that caused this error
        """
        self.source = source
        self.reason = reason
        message = f"Invalid audio source '{source}': {reason}"
        super().__init__(message, original_exception)


class TranscriptionError(AudioProcessingException):
    """Raised when transcription operation fails."""

    def __init__(
        self, 
        source: str, 
        reason: str, 
        transcript_id: str | None = None,
        original_exception: BaseException | None = None
    ):
        """
        Initialize the transcription error.

        :param source: Source that was being transcribed
        :param reason: Reason for the transcription failure
        :param transcript_id: Transcript ID if available
        :param original_exception: Original exception that caused this error
        """
        self.source = source
        self.reason = reason
        self.transcript_id = transcript_id
        message = f"Transcription failed for '{source}': {reason}"
        if transcript_id:
            message += f" (transcript_id: {transcript_id})"
        super().__init__(message, original_exception)


class AudioDownloadError(AudioProcessingException):
    """Raised when audio download operation fails."""

    def __init__(
        self, 
        url: str, 
        reason: str, 
        original_exception: BaseException | None = None
    ):
        """
        Initialize the audio download error.

        :param url: URL that failed to download
        :param reason: Reason for the download failure
        :param original_exception: Original exception that caused this error
        """
        self.url = url
        self.reason = reason
        message = f"Failed to download audio from '{url}': {reason}"
        super().__init__(message, original_exception)


class YouTubeVideoError(AudioProcessingException):
    """Raised when YouTube video processing fails."""

    def __init__(
        self, 
        video_id: str | None, 
        reason: str, 
        original_exception: BaseException | None = None
    ):
        """
        Initialize the YouTube video error.

        :param video_id: YouTube video ID that caused the error
        :param reason: Reason for the video processing failure
        :param original_exception: Original exception that caused this error
        """
        self.video_id = video_id
        self.reason = reason
        message = "YouTube video error"
        if video_id:
            message += f" for video '{video_id}'"
        message += f": {reason}"
        super().__init__(message, original_exception)


class ChunkingError(AudioProcessingException):
    """Raised when audio content chunking fails."""

    def __init__(
        self, 
        source: str, 
        reason: str, 
        original_exception: BaseException | None = None
    ):
        """
        Initialize the chunking error.

        :param source: Source being chunked
        :param reason: Reason for the chunking failure
        :param original_exception: Original exception that caused this error
        """
        self.source = source
        self.reason = reason
        message = f"Failed to chunk content from '{source}': {reason}"
        super().__init__(message, original_exception)


class ConfigurationLoadError(AudioProcessingException):
    """Raised when configuration file loading fails."""

    def __init__(
        self, 
        config_path: str, 
        reason: str, 
        original_exception: BaseException | None = None
    ):
        """
        Initialize the configuration load error.

        :param config_path: Path to the configuration file
        :param reason: Reason for the load failure
        :param original_exception: Original exception that caused this error
        """
        self.config_path = config_path
        self.reason = reason
        message = f"Failed to load configuration from '{config_path}': {reason}"
        super().__init__(message, original_exception)
