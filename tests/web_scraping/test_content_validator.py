"""
Tests for Content Validator.

Tests the content validation functionality including length checks,
word count validation, and content quality checks.
"""

from typing import Dict
from typing import Any

import pytest

from src.web_scraping.config import ContentValidationConfig
from src.web_scraping.validation.content_validator import DefaultContentValidator


class TestDefaultContentValidator:
    """Tests for DefaultContentValidator implementation."""

    @pytest.fixture
    def default_config(self) -> ContentValidationConfig:
        """Create default validation configuration."""
        return ContentValidationConfig()

    @pytest.fixture
    def custom_config(self) -> ContentValidationConfig:
        """Create custom validation configuration."""
        return ContentValidationConfig(
            min_content_length=50,
            max_content_length=1000,
            min_word_count=5,
            check_content_quality=True,
        )

    @pytest.fixture
    def no_quality_config(self) -> ContentValidationConfig:
        """Create configuration with quality checks disabled."""
        return ContentValidationConfig(
            min_content_length=10,
            max_content_length=10000,
            min_word_count=2,
            check_content_quality=False,
        )

    @pytest.fixture
    def default_validator(self, default_config: ContentValidationConfig) -> DefaultContentValidator:
        """Create validator with default config."""
        return DefaultContentValidator(default_config)

    @pytest.fixture
    def custom_validator(self, custom_config: ContentValidationConfig) -> DefaultContentValidator:
        """Create validator with custom config."""
        return DefaultContentValidator(custom_config)

    @pytest.fixture
    def no_quality_validator(self, no_quality_config: ContentValidationConfig) -> DefaultContentValidator:
        """Create validator with quality checks disabled."""
        return DefaultContentValidator(no_quality_config)

    @pytest.fixture
    def valid_content(self) -> str:
        """Create valid content for testing."""
        return "This is a valid content with enough words and sufficient length. " * 10

    @pytest.fixture
    def metadata(self) -> Dict[str, Any]:
        """Create sample metadata."""
        return {"url": "https://example.com/page"}


class TestValidateMethod(TestDefaultContentValidator):
    """Tests for the validate method."""

    def test_valid_content_passes(
        self,
        default_validator: DefaultContentValidator,
        valid_content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that valid content passes validation."""
        assert default_validator.validate(valid_content, metadata) is True

    def test_content_too_short_fails(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that content shorter than minimum fails."""
        short_content = "Too short"
        assert default_validator.validate(short_content, metadata) is False

    def test_content_too_long_fails(
        self,
        custom_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that content longer than maximum fails."""
        long_content = "x" * 2000  # Exceeds max of 1000
        assert custom_validator.validate(long_content, metadata) is False

    def test_too_few_words_fails(
        self,
        custom_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that content with too few words fails."""
        # Long enough but too few words (one long word)
        few_words = "x" * 100
        assert custom_validator.validate(few_words, metadata) is False

    def test_none_content_fails(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that None content fails validation."""
        assert default_validator.validate(None, metadata) is False

    def test_empty_content_fails(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that empty content fails validation."""
        assert default_validator.validate("", metadata) is False


class TestValidateWithErrors(TestDefaultContentValidator):
    """Tests for the validate_with_errors method."""

    def test_valid_content_returns_no_errors(
        self,
        default_validator: DefaultContentValidator,
        valid_content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that valid content returns no errors."""
        is_valid, errors = default_validator.validate_with_errors(valid_content, metadata)
        
        assert is_valid is True
        assert errors == []

    def test_none_content_returns_error(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that None content returns appropriate error."""
        is_valid, errors = default_validator.validate_with_errors(None, metadata)
        
        assert is_valid is False
        assert any("none" in error.lower() for error in errors)

    def test_short_content_returns_length_error(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that short content returns length error."""
        short_content = "Too short"
        is_valid, errors = default_validator.validate_with_errors(short_content, metadata)
        
        assert is_valid is False
        assert any("length" in error.lower() for error in errors)

    def test_long_content_returns_length_error(
        self,
        custom_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that long content returns length error."""
        long_content = "word " * 300  # Exceeds max of 1000
        is_valid, errors = custom_validator.validate_with_errors(long_content, metadata)
        
        assert is_valid is False
        assert any("exceeds maximum" in error.lower() for error in errors)

    def test_few_words_returns_word_count_error(
        self,
        custom_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that content with few words returns word count error."""
        # Content meeting length but not word count
        few_words = "x" * 100
        is_valid, errors = custom_validator.validate_with_errors(few_words, metadata)
        
        assert is_valid is False
        assert any("word count" in error.lower() for error in errors)

    def test_multiple_errors_returned(
        self,
        custom_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that multiple validation failures return multiple errors."""
        # Too short, too few words
        bad_content = "ab"
        is_valid, errors = custom_validator.validate_with_errors(bad_content, metadata)
        
        assert is_valid is False
        assert len(errors) >= 2


class TestContentQualityChecks(TestDefaultContentValidator):
    """Tests for content quality checking."""

    def test_mostly_whitespace_fails(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that content that is mostly whitespace fails."""
        # Content that is mostly spaces
        whitespace_content = "a" * 50 + " " * 500
        is_valid, errors = default_validator.validate_with_errors(whitespace_content, metadata)
        
        assert is_valid is False
        assert any("whitespace" in error.lower() for error in errors)

    def test_error_page_content_fails(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that content resembling error pages fails."""
        error_page = "404 not found - The page you requested could not be found"
        is_valid, errors = default_validator.validate_with_errors(error_page, metadata)
        
        assert is_valid is False
        assert any("error page" in error.lower() for error in errors)

    def test_403_forbidden_content_fails(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that 403 forbidden content fails."""
        forbidden_content = "403 Forbidden - Access Denied to this resource"
        is_valid, errors = default_validator.validate_with_errors(forbidden_content, metadata)
        
        assert is_valid is False
        assert any("error page" in error.lower() for error in errors)

    def test_quality_checks_disabled(
        self,
        no_quality_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that quality checks can be disabled."""
        # This would normally fail quality checks but should pass
        error_page = "404 not found but enough characters to pass length check"
        is_valid, errors = no_quality_validator.validate_with_errors(error_page, metadata)
        
        assert is_valid is True
        assert errors == []

    def test_repetitive_content_fails(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that highly repetitive content fails."""
        # Create repetitive content
        repetitive = ("spam spam spam spam spam " * 50)
        is_valid, errors = default_validator.validate_with_errors(repetitive, metadata)
        
        assert is_valid is False
        assert any("repetitive" in error.lower() for error in errors)


class TestEdgeCases(TestDefaultContentValidator):
    """Tests for edge cases."""

    def test_exact_minimum_length_passes(
        self,
        custom_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test content at exact minimum length passes."""
        # 50 chars minimum, 5 words minimum
        content = "word " * 10  # 50 chars, 10 words
        is_valid, _ = custom_validator.validate_with_errors(content, metadata)
        
        assert is_valid is True

    def test_exact_maximum_length_passes(
        self,
        no_quality_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test content at exact maximum length passes."""
        # no_quality_config has max_content_length=10000
        # Create content at exactly 10000 chars (non-repetitive)
        base = "This is sentence number %d. "
        content = ""
        i = 0
        while len(content) < 10000:
            content += base % i
            i += 1
        content = content[:10000]  # Truncate to exactly 10000
        is_valid, _ = no_quality_validator.validate_with_errors(content, metadata)
        
        assert is_valid is True

    def test_unicode_content_handled(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that unicode content is handled correctly."""
        unicode_content = "这是中文内容 " * 50 + "English content " * 50
        is_valid, _ = default_validator.validate_with_errors(unicode_content, metadata)
        
        # Should pass as it has enough length and words
        assert is_valid is True

    def test_multiline_content_handled(
        self,
        default_validator: DefaultContentValidator,
        metadata: Dict[str, Any],
    ) -> None:
        """Test that multiline content is handled correctly."""
        multiline_content = "Line one with content\n" * 20
        is_valid, _ = default_validator.validate_with_errors(multiline_content, metadata)
        
        assert is_valid is True

    def test_empty_metadata_handled(
        self,
        default_validator: DefaultContentValidator,
        valid_content: str,
    ) -> None:
        """Test that empty metadata is handled correctly."""
        is_valid, _ = default_validator.validate_with_errors(valid_content, {})
        
        assert is_valid is True
