"""
Tests for Metadata Enhancer.

This module tests the DefaultMetadataEnhancer class.
"""

import pytest

from src.youtube_processing.config import MetadataConfig
from src.youtube_processing.interfaces import VideoMetadata
from src.youtube_processing.metadata.enhancer import DefaultMetadataEnhancer


class TestDefaultMetadataEnhancer:
    """Tests for DefaultMetadataEnhancer class."""

    @pytest.fixture
    def config(self) -> MetadataConfig:
        """Create metadata config for testing."""
        return MetadataConfig(
            extract_description=True,
            extract_tags=True,
            extract_categories=True,
            extract_thumbnail_url=True,
            extract_view_count=True,
            extract_like_count=True,
            extract_channel_info=True,
            max_description_length=100,
        )

    @pytest.fixture
    def enhancer(self, config: MetadataConfig) -> DefaultMetadataEnhancer:
        """Create metadata enhancer for testing."""
        return DefaultMetadataEnhancer(config)

    @pytest.fixture
    def sample_metadata(self) -> VideoMetadata:
        """Create sample video metadata."""
        return VideoMetadata(
            video_id="test123",
            title="Test Video",
            description="This is a test video with a very long description that should be truncated when processed by the metadata enhancer.",
            channel_name="Test Channel",
            channel_id="UC123456",
            duration_seconds=180,
            upload_date="2024-01-15",
            view_count=1000000,
            like_count=50000,
            tags=["test", "video", "sample"],
            categories=["Education"],
            thumbnail_url="https://example.com/thumb.jpg",
        )

    class TestEnhance:
        """Tests for enhance method."""

        def test_enhance_truncates_long_description(
            self,
            enhancer: DefaultMetadataEnhancer,
            sample_metadata: VideoMetadata,
        ) -> None:
            """Test that long descriptions are truncated."""
            enhanced = enhancer.enhance(sample_metadata)

            assert len(enhanced.description) <= 103  # 100 + "..."
            assert enhanced.description.endswith("...")

        def test_enhance_preserves_short_description(
            self, config: MetadataConfig
        ) -> None:
            """Test that short descriptions are not truncated."""
            config = MetadataConfig(max_description_length=5000)
            enhancer = DefaultMetadataEnhancer(config)

            metadata = VideoMetadata(
                video_id="test123",
                title="Test",
                description="Short description",
            )

            enhanced = enhancer.enhance(metadata)
            assert enhanced.description == "Short description"

        def test_enhance_respects_extract_settings(
            self, sample_metadata: VideoMetadata
        ) -> None:
            """Test that extraction settings are respected."""
            config = MetadataConfig(
                extract_tags=False,
                extract_categories=False,
                extract_thumbnail_url=False,
            )
            enhancer = DefaultMetadataEnhancer(config)

            enhanced = enhancer.enhance(sample_metadata)

            assert enhanced.tags == []
            assert enhanced.categories == []
            assert enhanced.thumbnail_url == ""

        def test_enhance_adds_transcript_info_to_extra(
            self,
            enhancer: DefaultMetadataEnhancer,
            sample_metadata: VideoMetadata,
        ) -> None:
            """Test that transcript info is added to extra fields."""
            transcript = "This is the transcript text with many words."

            enhanced = enhancer.enhance(sample_metadata, transcript_text=transcript)

            assert "transcript_word_count" in enhanced.extra
            assert "transcript_char_count" in enhanced.extra
            assert enhanced.extra["transcript_word_count"] == 8

        def test_enhance_preserves_core_fields(
            self,
            enhancer: DefaultMetadataEnhancer,
            sample_metadata: VideoMetadata,
        ) -> None:
            """Test that core fields are preserved."""
            enhanced = enhancer.enhance(sample_metadata)

            assert enhanced.video_id == sample_metadata.video_id
            assert enhanced.title == sample_metadata.title
            assert enhanced.channel_name == sample_metadata.channel_name
            assert enhanced.duration_seconds == sample_metadata.duration_seconds
            assert enhanced.upload_date == sample_metadata.upload_date
            assert enhanced.view_count == sample_metadata.view_count
            assert enhanced.like_count == sample_metadata.like_count

    class TestMergeWithTranscriptMetadata:
        """Tests for merge_with_transcript_metadata method."""

        def test_merge_combines_metadata(
            self,
            enhancer: DefaultMetadataEnhancer,
            sample_metadata: VideoMetadata,
        ) -> None:
            """Test that video and transcript metadata are merged."""
            transcript_metadata = {
                "entities": ["Entity1", "Entity2"],
                "sentiment": "positive",
                "topics": ["Technology"],
            }

            merged = enhancer.merge_with_transcript_metadata(
                sample_metadata, transcript_metadata
            )

            # Video metadata should be present
            assert merged["video_id"] == "test123"
            assert merged["title"] == "Test Video"

            # Transcript metadata should be present
            assert merged["entities"] == ["Entity1", "Entity2"]
            assert merged["sentiment"] == "positive"

            # Source info should be added
            assert merged["source"] == "youtube"
            assert "source_url" in merged
            assert "processed_at" in merged

        def test_merge_namespaces_conflicts(
            self,
            enhancer: DefaultMetadataEnhancer,
        ) -> None:
            """Test that conflicting keys are namespaced."""
            metadata = VideoMetadata(
                video_id="test123",
                title="Test Video",
                language="en",
            )

            transcript_metadata = {
                "language": "detected_en",  # Conflicts with video language
            }

            merged = enhancer.merge_with_transcript_metadata(metadata, transcript_metadata)

            # Original language preserved
            assert merged["language"] == "en"
            # Transcript language namespaced
            assert merged["transcript_language"] == "detected_en"

        def test_merge_adds_source_url(
            self,
            enhancer: DefaultMetadataEnhancer,
        ) -> None:
            """Test that source URL is correctly formatted."""
            metadata = VideoMetadata(
                video_id="abc123xyz",
                title="Test",
            )

            merged = enhancer.merge_with_transcript_metadata(metadata, {})

            assert merged["source_url"] == "https://www.youtube.com/watch?v=abc123xyz"

        def test_merge_adds_timestamp(
            self,
            enhancer: DefaultMetadataEnhancer,
        ) -> None:
            """Test that processing timestamp is added."""
            metadata = VideoMetadata(
                video_id="test123",
                title="Test",
            )

            merged = enhancer.merge_with_transcript_metadata(metadata, {})

            assert "processed_at" in merged
            # Should be ISO format timestamp
            assert "T" in merged["processed_at"]
