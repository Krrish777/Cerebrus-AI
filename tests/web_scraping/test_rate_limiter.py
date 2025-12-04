"""
Tests for Rate Limiter.

Tests the token bucket rate limiting functionality.
"""

import time
from typing import Dict

import pytest

from src.web_scraping.config import RateLimitConfig
from src.web_scraping.rate_limiting.limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter implementation."""

    @pytest.fixture
    def default_config(self) -> RateLimitConfig:
        """Create default rate limit configuration."""
        return RateLimitConfig(
            enabled=True,
            requests_per_minute=60,  # 1 per second
            burst_size=5,
        )

    @pytest.fixture
    def fast_config(self) -> RateLimitConfig:
        """Create fast rate limit configuration for testing."""
        return RateLimitConfig(
            enabled=True,
            requests_per_minute=600,  # 10 per second
            burst_size=10,
        )

    @pytest.fixture
    def disabled_config(self) -> RateLimitConfig:
        """Create disabled rate limit configuration."""
        return RateLimitConfig(
            enabled=False,
            requests_per_minute=60,
            burst_size=5,
        )

    @pytest.fixture
    def default_limiter(self, default_config: RateLimitConfig) -> TokenBucketRateLimiter:
        """Create limiter with default config."""
        return TokenBucketRateLimiter(default_config)

    @pytest.fixture
    def fast_limiter(self, fast_config: RateLimitConfig) -> TokenBucketRateLimiter:
        """Create limiter with fast config."""
        return TokenBucketRateLimiter(fast_config)

    @pytest.fixture
    def disabled_limiter(self, disabled_config: RateLimitConfig) -> TokenBucketRateLimiter:
        """Create disabled limiter."""
        return TokenBucketRateLimiter(disabled_config)


class TestAcquireMethod(TestTokenBucketRateLimiter):
    """Tests for the acquire method."""

    def test_first_acquire_succeeds(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test that first acquire succeeds."""
        assert default_limiter.acquire("example.com") is True

    def test_burst_acquires_succeed(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test that burst size acquisitions succeed."""
        identifier = "burst.test.com"
        
        # Should be able to acquire burst_size times
        for i in range(5):
            assert default_limiter.acquire(identifier) is True, f"Acquire {i+1} should succeed"

    def test_exceeding_burst_fails(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test that exceeding burst size fails."""
        identifier = "exceed.test.com"
        
        # Exhaust burst
        for _ in range(5):
            default_limiter.acquire(identifier)
        
        # Next should fail
        assert default_limiter.acquire(identifier) is False

    def test_different_identifiers_independent(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test that different identifiers have independent limits."""
        # Exhaust burst for first identifier
        for _ in range(5):
            default_limiter.acquire("first.com")
        
        # Second identifier should still work
        assert default_limiter.acquire("second.com") is True

    def test_disabled_limiter_always_allows(self, disabled_limiter: TokenBucketRateLimiter) -> None:
        """Test that disabled limiter always allows."""
        identifier = "disabled.test.com"
        
        # Should always succeed
        for _ in range(100):
            assert disabled_limiter.acquire(identifier) is True

    def test_tokens_refill_over_time(self, fast_limiter: TokenBucketRateLimiter) -> None:
        """Test that tokens refill over time."""
        identifier = "refill.test.com"
        
        # Exhaust all tokens
        for _ in range(10):
            fast_limiter.acquire(identifier)
        
        # Should be exhausted
        assert fast_limiter.acquire(identifier) is False
        
        # Wait for refill (10 per second, so 0.15s should give at least 1 token)
        time.sleep(0.15)
        
        # Should have tokens again
        assert fast_limiter.acquire(identifier) is True


class TestAcquireBlockingMethod(TestTokenBucketRateLimiter):
    """Tests for the acquire_blocking method."""

    def test_blocking_acquire_immediate_success(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test that blocking acquire succeeds immediately when tokens available."""
        assert default_limiter.acquire_blocking("immediate.com", timeout=1.0) is True

    def test_blocking_acquire_waits_for_token(self, fast_limiter: TokenBucketRateLimiter) -> None:
        """Test that blocking acquire waits for token."""
        identifier = "wait.test.com"
        
        # Exhaust tokens
        for _ in range(10):
            fast_limiter.acquire(identifier)
        
        start = time.time()
        result = fast_limiter.acquire_blocking(identifier, timeout=1.0)
        elapsed = time.time() - start
        
        assert result is True
        assert elapsed > 0.05  # Should have waited

    def test_blocking_acquire_timeout(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test that blocking acquire respects timeout."""
        identifier = "timeout.test.com"
        
        # Exhaust tokens
        for _ in range(5):
            default_limiter.acquire(identifier)
        
        start = time.time()
        result = default_limiter.acquire_blocking(identifier, timeout=0.1)
        elapsed = time.time() - start
        
        # With 60 rpm (1 per second), 0.1s timeout should fail
        assert result is False
        assert elapsed >= 0.1
        assert elapsed < 0.3  # Should not wait much longer than timeout

    def test_disabled_blocking_always_succeeds(self, disabled_limiter: TokenBucketRateLimiter) -> None:
        """Test that disabled limiter always succeeds immediately."""
        start = time.time()
        result = disabled_limiter.acquire_blocking("disabled.com", timeout=1.0)
        elapsed = time.time() - start
        
        assert result is True
        assert elapsed < 0.1  # Should be immediate


class TestTimeUntilAvailable(TestTokenBucketRateLimiter):
    """Tests for the time_until_available method."""

    def test_immediate_availability(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test that new bucket shows immediate availability."""
        wait_time = default_limiter.time_until_available("new.com")
        
        assert wait_time == 0.0

    def test_after_burst_shows_wait_time(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test that exhausted bucket shows wait time."""
        identifier = "exhausted.com"
        
        # Exhaust tokens
        for _ in range(5):
            default_limiter.acquire(identifier)
        
        wait_time = default_limiter.time_until_available(identifier)
        
        # Should need to wait for 1 token at 60 rpm = 1 second
        assert wait_time > 0.5
        assert wait_time <= 2.0

    def test_disabled_shows_zero_wait(self, disabled_limiter: TokenBucketRateLimiter) -> None:
        """Test that disabled limiter always shows zero wait."""
        wait_time = disabled_limiter.time_until_available("any.com")
        
        assert wait_time == 0.0


class TestResetMethods(TestTokenBucketRateLimiter):
    """Tests for reset methods."""

    def test_reset_single_identifier(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test resetting a single identifier."""
        identifier = "reset.test.com"
        
        # Exhaust tokens
        for _ in range(5):
            default_limiter.acquire(identifier)
        
        # Should be exhausted
        assert default_limiter.acquire(identifier) is False
        
        # Reset
        default_limiter.reset(identifier)
        
        # Should work again
        assert default_limiter.acquire(identifier) is True

    def test_reset_does_not_affect_others(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test that reset doesn't affect other identifiers."""
        # Exhaust both
        for _ in range(5):
            default_limiter.acquire("first.com")
            default_limiter.acquire("second.com")
        
        # Reset only first
        default_limiter.reset("first.com")
        
        # First should work, second should not
        assert default_limiter.acquire("first.com") is True
        assert default_limiter.acquire("second.com") is False

    def test_reset_all(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test resetting all identifiers."""
        # Exhaust multiple
        for _ in range(5):
            default_limiter.acquire("a.com")
            default_limiter.acquire("b.com")
            default_limiter.acquire("c.com")
        
        # All exhausted
        assert default_limiter.acquire("a.com") is False
        assert default_limiter.acquire("b.com") is False
        assert default_limiter.acquire("c.com") is False
        
        # Reset all
        default_limiter.reset_all()
        
        # All should work
        assert default_limiter.acquire("a.com") is True
        assert default_limiter.acquire("b.com") is True
        assert default_limiter.acquire("c.com") is True


class TestBucketStatus(TestTokenBucketRateLimiter):
    """Tests for get_bucket_status method."""

    def test_new_bucket_status(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test status of new bucket."""
        status = default_limiter.get_bucket_status("new.com")
        
        assert status["tokens"] == 5.0
        assert status["capacity"] == 5
        assert status["utilization"] == 0.0

    def test_partially_used_bucket_status(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test status of partially used bucket."""
        identifier = "partial.com"
        
        # Use 3 tokens
        for _ in range(3):
            default_limiter.acquire(identifier)
        
        status = default_limiter.get_bucket_status(identifier)
        
        assert status["tokens"] == pytest.approx(2.0, abs=0.5)
        assert status["capacity"] == 5
        assert status["utilization"] > 0.5

    def test_exhausted_bucket_status(self, default_limiter: TokenBucketRateLimiter) -> None:
        """Test status of exhausted bucket."""
        identifier = "exhausted.com"
        
        # Exhaust all tokens
        for _ in range(5):
            default_limiter.acquire(identifier)
        
        status = default_limiter.get_bucket_status(identifier)
        
        assert status["tokens"] < 1.0
        assert status["utilization"] > 0.8
