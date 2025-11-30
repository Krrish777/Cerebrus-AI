"""
Unit tests for data extractors.

Tests all extractor implementations following AGENTS.md testing standards.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.audio_processing.extractors.base import BaseExtractor
from src.audio_processing.extractors.chapter import ChapterExtractor
from src.audio_processing.extractors.content_safety import ContentSafetyExtractor
from src.audio_processing.extractors.entity import EntityExtractor
from src.audio_processing.extractors.highlights import HighlightsExtractor
from src.audio_processing.extractors.registry import ExtractorRegistry, get_registry
from src.audio_processing.extractors.sentiment import SentimentExtractor
from src.audio_processing.extractors.topic import TopicExtractor
from src.audio_processing.exceptions import ExtractionError


@pytest.fixture
def mock_transcript_data() -> Dict[str, Any]:
    """Load mock transcript data from fixture."""
    fixture_path = Path("data") / "fixtures" / "mock_transcript_response.json"
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))

    # Fallback inline data
    return {
        "id": "test-123",
        "text": "Hello world. This is a test transcript.",
        "confidence": 0.95,
        "chapters": [
            {
                "headline": "Introduction",
                "summary": "The speaker introduces the topic",
                "gist": "intro",
                "start": 0,
                "end": 60000,
            },
            {
                "headline": "Main Content",
                "summary": "The main discussion",
                "gist": "main",
                "start": 60000,
                "end": 180000,
            },
        ],
    }


@pytest.fixture
def mock_analysis_data() -> Dict[str, Any]:
    """Load mock analysis data from fixture."""
    fixture_path = Path("data") / "fixtures" / "mock_analysis_data.json"
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))

    # Fallback inline data
    return {
        "sentiment_analysis": [
            {
                "text": "This is great!",
                "sentiment": "POSITIVE",
                "confidence": 0.95,
                "start": 1000,
                "end": 2500,
            },
            {
                "text": "I don't like this.",
                "sentiment": "NEGATIVE",
                "confidence": 0.88,
                "start": 3000,
                "end": 4500,
            },
        ],
        "entities": [
            {
                "text": "John Smith",
                "entity_type": "PERSON",
                "start": 1000,
                "end": 1500,
            },
            {
                "text": "Acme Corp",
                "entity_type": "ORGANIZATION",
                "start": 2000,
                "end": 2500,
            },
        ],
        "iab_categories": {
            "status": "success",
            "results": [
                {
                    "text": "Machine learning is transforming...",
                    "labels": [
                        {
                            "label": "Technology&Computing>ArtificialIntelligence",
                            "relevance": 0.95,
                        }
                    ],
                }
            ],
            "summary": {
                "Technology&Computing>ArtificialIntelligence": 0.95,
                "BusinessAndFinance>Technology": 0.7,
            },
        },
        "content_safety": {
            "status": "success",
            "results": [],
            "summary": {},
        },
        "auto_highlights": {
            "status": "success",
            "results": [
                {
                    "text": "machine learning",
                    "count": 5,
                    "rank": 0.95,
                    "timestamps": [
                        {"start": 1000, "end": 1500},
                        {"start": 5000, "end": 5500},
                    ],
                }
            ],
        },
    }


class TestSentimentExtractor:
    """Test suite for SentimentExtractor."""

    def test_extract_returns_structured_data(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test that extract returns properly structured data."""
        extractor = SentimentExtractor()
        result = extractor.extract(mock_analysis_data)

        assert "items" in result
        assert "count" in result
        assert "distribution" in result
        assert "dominant_sentiment" in result
        assert "average_confidence" in result

    def test_extract_calculates_distribution(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test that distribution is calculated correctly."""
        extractor = SentimentExtractor()
        result = extractor.extract(mock_analysis_data)

        # Fixture has 2 NEUTRAL and 1 POSITIVE
        assert result["distribution"]["POSITIVE"] >= 1
        assert result["distribution"]["NEUTRAL"] >= 1

    def test_is_available_returns_false_for_missing_data(self) -> None:
        """Test is_available returns false when data is missing."""
        extractor = SentimentExtractor()
        result = extractor.is_available({})

        assert result is False

    def test_is_available_returns_true_for_present_data(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test is_available returns true when data is present."""
        extractor = SentimentExtractor()
        result = extractor.is_available(mock_analysis_data)

        assert result is True

    def test_filter_by_sentiment(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test filtering by sentiment type."""
        extractor = SentimentExtractor()
        positive = extractor.filter_by_sentiment(mock_analysis_data, "POSITIVE")

        assert len(positive) >= 1
        assert all(item["sentiment"] == "POSITIVE" for item in positive)

    def test_normalize_sentiment(self) -> None:
        """Test sentiment normalization."""
        extractor = SentimentExtractor()

        assert extractor._normalize_sentiment("positive") == "POSITIVE"
        assert extractor._normalize_sentiment("NEGATIVE") == "NEGATIVE"
        assert extractor._normalize_sentiment("unknown") == "NEUTRAL"


class TestEntityExtractor:
    """Test suite for EntityExtractor."""

    def test_extract_returns_structured_data(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test that extract returns properly structured data."""
        extractor = EntityExtractor()
        result = extractor.extract(mock_analysis_data)

        assert "items" in result
        assert "count" in result
        assert "by_type" in result
        assert "unique_entities" in result

    def test_extract_groups_by_type(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test that entities are grouped by type."""
        extractor = EntityExtractor()
        result = extractor.extract(mock_analysis_data)

        # Fixture has TOPIC entities (technology, innovation)
        assert len(result["by_type"]) > 0
        assert result["unique_entities"] > 0

    def test_get_entities_by_type(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test filtering entities by type."""
        extractor = EntityExtractor()

        # Get the actual types from the fixture
        result = extractor.extract(mock_analysis_data)
        if result["entity_types"]:
            entity_type = result["entity_types"][0]
            filtered = extractor.get_entities_by_type(mock_analysis_data, entity_type)
            assert len(filtered) >= 1

    def test_normalize_entity_type(self) -> None:
        """Test entity type normalization."""
        extractor = EntityExtractor()

        assert extractor._normalize_entity_type("PERSON_NAME") == "PERSON"
        assert extractor._normalize_entity_type("ORG") == "ORGANIZATION"
        assert extractor._normalize_entity_type("GPE") == "LOCATION"


class TestChapterExtractor:
    """Test suite for ChapterExtractor."""

    def test_extract_returns_structured_data(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test that extract returns properly structured data."""
        extractor = ChapterExtractor()
        result = extractor.extract(mock_transcript_data)

        assert "items" in result
        assert "count" in result
        assert "total_duration" in result

    def test_extract_calculates_duration(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test that duration is calculated correctly."""
        extractor = ChapterExtractor()
        result = extractor.extract(mock_transcript_data)

        # Each chapter should have duration
        for item in result["items"]:
            assert "duration" in item
            assert item["duration"] >= 0

    def test_get_chapter_at_time(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test getting chapter at specific time."""
        extractor = ChapterExtractor()
        # Fixture has chapter from 0-5000 and 5001-14000
        chapter = extractor.get_chapter_at_time(mock_transcript_data, 2000)

        assert chapter is not None
        assert chapter["headline"] == "Introduction"

    def test_get_headlines(
        self,
        mock_transcript_data: Dict[str, Any],
    ) -> None:
        """Test getting all headlines."""
        extractor = ChapterExtractor()
        headlines = extractor.get_headlines(mock_transcript_data)

        assert len(headlines) == 2
        assert "Introduction" in headlines

    def test_format_timestamp(self) -> None:
        """Test timestamp formatting."""
        extractor = ChapterExtractor()

        assert extractor.format_timestamp(65000) == "01:05"
        assert extractor.format_timestamp(3665000) == "01:01:05"


class TestTopicExtractor:
    """Test suite for TopicExtractor."""

    def test_extract_returns_structured_data(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test that extract returns properly structured data."""
        extractor = TopicExtractor()
        result = extractor.extract(mock_analysis_data)

        assert "items" in result
        assert "count" in result
        assert "summary" in result
        assert "top_topics" in result

    def test_parse_iab_label(self) -> None:
        """Test IAB label parsing."""
        extractor = TopicExtractor()

        category, subcategory = extractor._parse_iab_label(
            "Technology&Computing>ArtificialIntelligence"
        )

        assert category == "Technology&Computing"
        assert subcategory == "ArtificialIntelligence"

    def test_get_top_topics(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test getting top topics."""
        extractor = TopicExtractor()
        top = extractor.get_top_topics(mock_analysis_data, limit=1)

        assert len(top) == 1
        # Fixture has "Technology & Computing" with 0.75 as highest
        assert top[0]["relevance"] > 0


class TestContentSafetyExtractor:
    """Test suite for ContentSafetyExtractor."""

    def test_extract_safe_content(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test extracting safe content."""
        extractor = ContentSafetyExtractor()
        result = extractor.extract(mock_analysis_data)

        assert result["is_safe"] is True
        assert result["severity_score"] == 0.0

    def test_is_safe_returns_true_for_safe_content(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test is_safe method."""
        extractor = ContentSafetyExtractor()
        assert extractor.is_safe(mock_analysis_data) is True

    def test_extract_flagged_content(self) -> None:
        """Test extracting flagged content."""
        flagged_data = {
            "content_safety": {
                "results": [
                    {
                        "text": "Some flagged content",
                        "labels": [
                            {"label": "profanity", "confidence": 0.9, "severity": 0.6}
                        ],
                        "start": 1000,
                        "end": 2000,
                    }
                ],
                "summary": {"profanity": 0.6},
            }
        }

        extractor = ContentSafetyExtractor()
        result = extractor.extract(flagged_data)

        assert result["is_safe"] is False
        assert result["severity_score"] == 0.6
        assert "profanity" in result["flagged_labels"]


class TestHighlightsExtractor:
    """Test suite for HighlightsExtractor."""

    def test_extract_returns_structured_data(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test that extract returns properly structured data."""
        extractor = HighlightsExtractor()
        result = extractor.extract(mock_analysis_data)

        assert "items" in result
        assert "count" in result
        assert "top_highlights" in result
        assert "total_occurrences" in result

    def test_get_top_highlights(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test getting top highlights."""
        extractor = HighlightsExtractor()
        top = extractor.get_top_highlights(mock_analysis_data, limit=1)

        assert len(top) == 1
        # Fixture has "technology and innovation" as top highlight
        assert len(top[0]["text"]) > 0

    def test_search_highlights(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test searching highlights."""
        extractor = HighlightsExtractor()
        # Fixture has "technology and innovation" 
        results = extractor.search_highlights(mock_analysis_data, "technology")

        assert len(results) >= 1
        assert "technology" in results[0]["text"].lower()


class TestExtractorRegistry:
    """Test suite for ExtractorRegistry."""

    def test_available_returns_all_extractors(self) -> None:
        """Test that available returns all registered extractors."""
        registry = ExtractorRegistry()
        available = registry.available()

        assert "sentiment" in available
        assert "entity" in available
        assert "chapter" in available
        assert "topic" in available
        assert "content_safety" in available
        assert "highlights" in available

    def test_get_returns_extractor_instance(self) -> None:
        """Test that get returns an extractor instance."""
        registry = ExtractorRegistry()
        extractor = registry.get("sentiment")

        assert isinstance(extractor, SentimentExtractor)

    def test_get_raises_for_unknown_extractor(self) -> None:
        """Test that get raises for unknown extractor."""
        from src.audio_processing.exceptions import ConfigurationError

        registry = ExtractorRegistry()

        with pytest.raises(ConfigurationError):
            registry.get("unknown")

    def test_extract_all(
        self,
        mock_analysis_data: Dict[str, Any],
    ) -> None:
        """Test extracting with all extractors."""
        registry = ExtractorRegistry()

        # Add chapter data for complete test
        mock_analysis_data["chapters"] = [
            {
                "headline": "Test",
                "summary": "Test summary",
                "gist": "test",
                "start": 0,
                "end": 60000,
            }
        ]

        results = registry.extract_all(mock_analysis_data)

        # Should have results for extractors with available data
        assert len(results) > 0

    def test_register_custom_extractor(self) -> None:
        """Test registering a custom extractor."""

        class CustomExtractor(BaseExtractor):
            @property
            def extractor_name(self) -> str:
                return "custom"

            def _get_data_key(self) -> str:
                return "custom_data"

            def _do_extract(self, data):
                return {"items": [], "count": 0}

        registry = ExtractorRegistry()
        registry.register("custom", CustomExtractor)

        assert "custom" in registry.available()

    def test_get_global_registry(self) -> None:
        """Test getting global registry."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2
