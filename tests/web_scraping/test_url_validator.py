"""
Tests for URL Validator.

Tests the URL validation functionality including scheme validation,
domain blocking, TLD checking, and URL normalization.
"""

from typing import Tuple

import pytest

from src.web_scraping.config import URLValidationConfig
from src.web_scraping.exceptions import URLValidationError
from src.web_scraping.validation.url_validator import DefaultURLValidator


class TestDefaultURLValidator:
    """Tests for DefaultURLValidator implementation."""

    @pytest.fixture
    def default_config(self) -> URLValidationConfig:
        """Create default validation configuration."""
        return URLValidationConfig()

    @pytest.fixture
    def strict_config(self) -> URLValidationConfig:
        """Create strict validation configuration."""
        return URLValidationConfig(
            allowed_schemes=("https",),
            blocked_domains=("blocked.com", "spam.net"),
            max_url_length=100,
            require_valid_tld=True,
        )

    @pytest.fixture
    def default_validator(self, default_config: URLValidationConfig) -> DefaultURLValidator:
        """Create validator with default config."""
        return DefaultURLValidator(default_config)

    @pytest.fixture
    def strict_validator(self, strict_config: URLValidationConfig) -> DefaultURLValidator:
        """Create validator with strict config."""
        return DefaultURLValidator(strict_config)


class TestValidateMethod(TestDefaultURLValidator):
    """Tests for the validate method."""

    def test_valid_https_url(self, default_validator: DefaultURLValidator) -> None:
        """Test validation of valid HTTPS URL."""
        assert default_validator.validate("https://example.com") is True

    def test_valid_http_url(self, default_validator: DefaultURLValidator) -> None:
        """Test validation of valid HTTP URL."""
        assert default_validator.validate("http://example.com") is True

    def test_valid_url_with_path(self, default_validator: DefaultURLValidator) -> None:
        """Test validation of URL with path."""
        assert default_validator.validate("https://example.com/path/to/page") is True

    def test_valid_url_with_query(self, default_validator: DefaultURLValidator) -> None:
        """Test validation of URL with query string."""
        assert default_validator.validate("https://example.com?query=value") is True

    def test_empty_url_invalid(self, default_validator: DefaultURLValidator) -> None:
        """Test that empty URL is invalid."""
        assert default_validator.validate("") is False

    def test_url_without_scheme_invalid(self, default_validator: DefaultURLValidator) -> None:
        """Test that URL without scheme is invalid."""
        assert default_validator.validate("example.com") is False

    def test_invalid_scheme(self, default_validator: DefaultURLValidator) -> None:
        """Test that invalid scheme is rejected."""
        assert default_validator.validate("ftp://example.com") is False

    def test_https_only_rejects_http(self, strict_validator: DefaultURLValidator) -> None:
        """Test that HTTPS-only config rejects HTTP."""
        assert strict_validator.validate("http://example.com") is False

    def test_blocked_domain_rejected(self, strict_validator: DefaultURLValidator) -> None:
        """Test that blocked domains are rejected."""
        assert strict_validator.validate("https://blocked.com") is False
        assert strict_validator.validate("https://spam.net/page") is False

    def test_subdomain_of_blocked_domain_rejected(self, strict_validator: DefaultURLValidator) -> None:
        """Test that subdomains of blocked domains are rejected."""
        assert strict_validator.validate("https://sub.blocked.com") is False
        assert strict_validator.validate("https://www.spam.net") is False

    def test_url_too_long_rejected(self, strict_validator: DefaultURLValidator) -> None:
        """Test that URLs exceeding max length are rejected."""
        long_url = "https://example.com/" + "a" * 100
        assert strict_validator.validate(long_url) is False


class TestValidateWithErrors(TestDefaultURLValidator):
    """Tests for the validate_with_errors method."""

    def test_valid_url_returns_no_errors(self, default_validator: DefaultURLValidator) -> None:
        """Test that valid URL returns no errors."""
        is_valid, errors = default_validator.validate_with_errors("https://example.com")
        
        assert is_valid is True
        assert errors == []

    def test_empty_url_returns_error(self, default_validator: DefaultURLValidator) -> None:
        """Test that empty URL returns appropriate error."""
        is_valid, errors = default_validator.validate_with_errors("")
        
        assert is_valid is False
        assert "URL cannot be empty" in errors

    def test_missing_scheme_returns_error(self, default_validator: DefaultURLValidator) -> None:
        """Test that missing scheme returns appropriate error."""
        is_valid, errors = default_validator.validate_with_errors("example.com")
        
        assert is_valid is False
        assert any("scheme" in error.lower() for error in errors)

    def test_invalid_scheme_returns_error(self, default_validator: DefaultURLValidator) -> None:
        """Test that invalid scheme returns appropriate error."""
        is_valid, errors = default_validator.validate_with_errors("ftp://example.com")
        
        assert is_valid is False
        assert any("scheme" in error.lower() and "ftp" in error.lower() for error in errors)

    def test_blocked_domain_returns_error(self, strict_validator: DefaultURLValidator) -> None:
        """Test that blocked domain returns appropriate error."""
        is_valid, errors = strict_validator.validate_with_errors("https://blocked.com")
        
        assert is_valid is False
        assert any("blocked" in error.lower() for error in errors)

    def test_url_too_long_returns_error(self, strict_validator: DefaultURLValidator) -> None:
        """Test that long URL returns appropriate error."""
        long_url = "https://example.com/" + "a" * 100
        is_valid, errors = strict_validator.validate_with_errors(long_url)
        
        assert is_valid is False
        assert any("length" in error.lower() for error in errors)

    def test_multiple_errors_returned(self, strict_validator: DefaultURLValidator) -> None:
        """Test that multiple validation failures return multiple errors."""
        # HTTP (not allowed) + blocked domain + too long
        long_blocked_url = "http://blocked.com/" + "a" * 100
        is_valid, errors = strict_validator.validate_with_errors(long_blocked_url)
        
        assert is_valid is False
        assert len(errors) >= 2


class TestNormalizeUrl(TestDefaultURLValidator):
    """Tests for the normalize_url method."""

    def test_lowercase_scheme(self, default_validator: DefaultURLValidator) -> None:
        """Test that scheme is lowercased."""
        normalized = default_validator.normalize_url("HTTPS://example.com")
        
        assert normalized.startswith("https://")

    def test_lowercase_host(self, default_validator: DefaultURLValidator) -> None:
        """Test that host is lowercased."""
        normalized = default_validator.normalize_url("https://EXAMPLE.COM")
        
        assert "example.com" in normalized

    def test_remove_default_https_port(self, default_validator: DefaultURLValidator) -> None:
        """Test that default HTTPS port is removed."""
        normalized = default_validator.normalize_url("https://example.com:443/path")
        
        assert ":443" not in normalized
        assert "example.com/path" in normalized

    def test_remove_default_http_port(self, default_validator: DefaultURLValidator) -> None:
        """Test that default HTTP port is removed."""
        normalized = default_validator.normalize_url("http://example.com:80/path")
        
        assert ":80" not in normalized
        assert "example.com/path" in normalized

    def test_keep_non_default_port(self, default_validator: DefaultURLValidator) -> None:
        """Test that non-default ports are kept."""
        normalized = default_validator.normalize_url("https://example.com:8443/path")
        
        assert ":8443" in normalized

    def test_remove_trailing_slash(self, default_validator: DefaultURLValidator) -> None:
        """Test that trailing slashes are removed from path."""
        normalized = default_validator.normalize_url("https://example.com/path/")
        
        assert normalized.endswith("/path")

    def test_keep_root_path(self, default_validator: DefaultURLValidator) -> None:
        """Test that root path is kept."""
        normalized = default_validator.normalize_url("https://example.com")
        
        assert normalized == "https://example.com/"

    def test_preserve_query_string(self, default_validator: DefaultURLValidator) -> None:
        """Test that query string is preserved."""
        normalized = default_validator.normalize_url("https://example.com/path?foo=bar")
        
        assert "?foo=bar" in normalized

    def test_preserve_fragment(self, default_validator: DefaultURLValidator) -> None:
        """Test that fragment is preserved."""
        normalized = default_validator.normalize_url("https://example.com/path#section")
        
        assert "#section" in normalized

    def test_empty_url_raises_error(self, default_validator: DefaultURLValidator) -> None:
        """Test that empty URL raises URLValidationError."""
        with pytest.raises(URLValidationError):
            default_validator.normalize_url("")


class TestExtractDomain(TestDefaultURLValidator):
    """Tests for the extract_domain method."""

    def test_simple_domain(self, default_validator: DefaultURLValidator) -> None:
        """Test extracting simple domain."""
        domain = default_validator.extract_domain("https://example.com")
        
        assert domain == "example.com"

    def test_domain_with_subdomain(self, default_validator: DefaultURLValidator) -> None:
        """Test extracting domain with subdomain."""
        domain = default_validator.extract_domain("https://www.example.com")
        
        assert domain == "www.example.com"

    def test_domain_with_port(self, default_validator: DefaultURLValidator) -> None:
        """Test extracting domain with port (port should be removed)."""
        domain = default_validator.extract_domain("https://example.com:8080")
        
        assert domain == "example.com"

    def test_domain_with_path(self, default_validator: DefaultURLValidator) -> None:
        """Test extracting domain from URL with path."""
        domain = default_validator.extract_domain("https://example.com/path/to/page")
        
        assert domain == "example.com"

    def test_domain_with_credentials(self, default_validator: DefaultURLValidator) -> None:
        """Test extracting domain from URL with credentials."""
        domain = default_validator.extract_domain("https://user:pass@example.com")
        
        assert domain == "example.com"

    def test_lowercase_domain(self, default_validator: DefaultURLValidator) -> None:
        """Test that domain is lowercased."""
        domain = default_validator.extract_domain("https://EXAMPLE.COM")
        
        assert domain == "example.com"

    def test_invalid_url_returns_empty(self, default_validator: DefaultURLValidator) -> None:
        """Test that invalid URL returns empty string."""
        domain = default_validator.extract_domain("")
        
        assert domain == ""


class TestTLDValidation(TestDefaultURLValidator):
    """Tests for TLD validation."""

    def test_valid_common_tlds(self, default_validator: DefaultURLValidator) -> None:
        """Test validation of common TLDs."""
        valid_tlds = ["com", "org", "net", "edu", "gov", "io", "ai", "dev"]
        
        for tld in valid_tlds:
            url = f"https://example.{tld}"
            assert default_validator.validate(url) is True, f"TLD '{tld}' should be valid"

    def test_valid_country_tlds(self, default_validator: DefaultURLValidator) -> None:
        """Test validation of country TLDs."""
        valid_tlds = ["uk", "de", "fr", "jp", "au", "ca"]
        
        for tld in valid_tlds:
            url = f"https://example.{tld}"
            assert default_validator.validate(url) is True, f"Country TLD '{tld}' should be valid"

    def test_invalid_tld_rejected_when_required(self, default_validator: DefaultURLValidator) -> None:
        """Test that invalid TLD is rejected when require_valid_tld is True."""
        is_valid, errors = default_validator.validate_with_errors("https://example.invalidtld")
        
        assert is_valid is False
        assert any("tld" in error.lower() for error in errors)

    def test_tld_validation_disabled(self) -> None:
        """Test that TLD validation can be disabled."""
        config = URLValidationConfig(require_valid_tld=False)
        validator = DefaultURLValidator(config)
        
        assert validator.validate("https://example.invalidtld") is True
