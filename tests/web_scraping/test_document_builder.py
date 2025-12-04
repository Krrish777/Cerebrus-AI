"""
Tests for Document Builder.

Tests the document building functionality including metadata handling,
content processing, and Haystack Document creation.
"""

from datetime import datetime

import pytest

from src.web_scraping.processing.document_builder import DefaultDocumentBuilder
from src.web_scraping.interfaces import ScrapedContent


class TestDefaultDocumentBuilder:
    """Tests for DefaultDocumentBuilder implementation."""

    @pytest.fixture
    def builder(self) -> DefaultDocumentBuilder:
        """Create document builder instance."""
        return DefaultDocumentBuilder()

    @pytest.fixture
    def builder_without_links(self) -> DefaultDocumentBuilder:
        """Create document builder that excludes links."""
        return DefaultDocumentBuilder(include_links_in_meta=False)

    @pytest.fixture
    def sample_scraped_content(self) -> ScrapedContent:
        """Create sample scraped content."""
        return ScrapedContent(
            url="https://example.com/article",
            content="# Article Title\n\nThis is the article content.",
            title="Article Title",
            description="A sample article",
            links=["https://example.com/link1", "https://example.com/link2"],
            metadata={"author": "Test Author"},
            word_count=6,
        )


class TestBuildDocument(TestDefaultDocumentBuilder):
    """Tests for building documents from scraped content."""

    def test_build_from_scraped_content(
        self,
        builder: DefaultDocumentBuilder,
        sample_scraped_content: ScrapedContent,
    ) -> None:
        """Test building document from ScrapedContent."""
        document = builder.build(sample_scraped_content)
        
        assert document is not None
        assert document.content == sample_scraped_content.content
        assert document.meta["url"] == sample_scraped_content.url
        assert document.meta["title"] == sample_scraped_content.title

    def test_build_preserves_metadata(
        self,
        builder: DefaultDocumentBuilder,
        sample_scraped_content: ScrapedContent,
    ) -> None:
        """Test that building preserves metadata."""
        document = builder.build(sample_scraped_content)
        
        assert "author" in document.meta
        assert document.meta["author"] == "Test Author"

    def test_build_adds_default_metadata(
        self,
        builder: DefaultDocumentBuilder,
    ) -> None:
        """Test that building adds default metadata."""
        content = ScrapedContent(
            url="https://example.com",
            content="Simple content",
            title="Title",
            word_count=2,
        )
        
        document = builder.build(content)
        
        # Should have source URL and title
        assert document.meta["url"] == "https://example.com"
        assert document.meta["title"] == "Title"
        assert document.meta["source_type"] == "web"


class TestBuildBatch(TestDefaultDocumentBuilder):
    """Tests for batch document building."""

    def test_build_batch(self, builder: DefaultDocumentBuilder) -> None:
        """Test building multiple documents."""
        contents = [
            ScrapedContent(
                url=f"https://example.com/page{i}",
                content=f"Content for page {i}",
                title=f"Page {i}",
                word_count=4,
            )
            for i in range(3)
        ]
        
        documents = builder.build_batch(contents)
        
        assert len(documents) == 3
        for i, doc in enumerate(documents):
            assert doc.meta["url"] == f"https://example.com/page{i}"


class TestMetadataEnrichment(TestDefaultDocumentBuilder):
    """Tests for metadata enrichment."""

    def test_adds_scrape_timestamp(self, builder: DefaultDocumentBuilder) -> None:
        """Test that scrape timestamp is added to metadata."""
        content = ScrapedContent(
            url="https://example.com",
            content="Content",
            title="Title",
            word_count=1,
        )
        
        document = builder.build(content)
        
        # Should have timestamp or scraped_at metadata
        assert "scraped_at" in document.meta

    def test_adds_source_type(self, builder: DefaultDocumentBuilder) -> None:
        """Test that source type is added to metadata."""
        content = ScrapedContent(
            url="https://example.com",
            content="Content",
            title="Title",
            word_count=1,
        )
        
        document = builder.build(content)
        
        assert "source_type" in document.meta
        assert document.meta["source_type"] == "web"

    def test_includes_links_when_enabled(
        self,
        builder: DefaultDocumentBuilder,
        sample_scraped_content: ScrapedContent,
    ) -> None:
        """Test that links are included when enabled."""
        document = builder.build(sample_scraped_content)
        
        assert "links" in document.meta
        assert len(document.meta["links"]) == 2

    def test_excludes_links_when_disabled(
        self,
        builder_without_links: DefaultDocumentBuilder,
        sample_scraped_content: ScrapedContent,
    ) -> None:
        """Test that links are excluded when disabled."""
        document = builder_without_links.build(sample_scraped_content)
        
        # Links should not be in metadata when disabled
        assert "links" not in document.meta or document.meta.get("links") is None


class TestDocumentId(TestDefaultDocumentBuilder):
    """Tests for document ID generation."""

    def test_generates_document_id(self, builder: DefaultDocumentBuilder) -> None:
        """Test that document ID is generated."""
        content = ScrapedContent(
            url="https://example.com",
            content="Content",
            title="Title",
            word_count=1,
        )
        
        document = builder.build(content)
        
        assert document.id is not None
        assert len(document.id) > 0
        assert document.id.startswith("web_")


class TestEdgeCases(TestDefaultDocumentBuilder):
    """Tests for edge cases."""

    def test_special_characters_in_content(self, builder: DefaultDocumentBuilder) -> None:
        """Test handling of special characters."""
        content = ScrapedContent(
            url="https://example.com",
            content="<script>alert('xss')</script> & < > \"",
            title="Title <Test>",
            word_count=5,
        )
        
        document = builder.build(content)
        
        assert document is not None
        assert document.content is not None

    def test_unicode_content(self, builder: DefaultDocumentBuilder) -> None:
        """Test handling of unicode content."""
        content = ScrapedContent(
            url="https://example.com/日本語",
            content="日本語コンテンツ 中文内容 العربية",
            title="多言語タイトル",
            word_count=4,
        )
        
        document = builder.build(content)
        
        assert "日本語" in document.content
        assert document.meta["title"] == "多言語タイトル"

    def test_very_long_content(self, builder: DefaultDocumentBuilder) -> None:
        """Test handling of very long content."""
        long_content = "Word " * 100000  # 100K words
        
        content = ScrapedContent(
            url="https://example.com",
            content=long_content,
            title="Long Content",
            word_count=100000,
        )
        
        document = builder.build(content)
        
        assert len(document.content) > 0

    def test_empty_title_handling(self, builder: DefaultDocumentBuilder) -> None:
        """Test handling of empty title."""
        content = ScrapedContent(
            url="https://example.com",
            content="Content without title",
            title="",
            word_count=3,
        )
        
        document = builder.build(content)
        
        assert document is not None
        assert document.meta["title"] == ""

    def test_content_hash_generation(self, builder: DefaultDocumentBuilder) -> None:
        """Test that content hash is generated."""
        content = ScrapedContent(
            url="https://example.com",
            content="Test content for hashing",
            title="Title",
            word_count=4,
        )
        
        document = builder.build(content)
        
        assert "content_hash" in document.meta
        assert len(document.meta["content_hash"]) > 0
