"""
Topic extractor implementation.

Extracts IAB category/topic data from transcripts.
"""

from typing import Any, Dict, List, Tuple

from src.audio_processing.extractors.base import BaseExtractor
from src.core.logging import get_logger

logger = get_logger(__name__)


class TopicExtractor(BaseExtractor):
    """
    Extractor for IAB category/topic detection.

    Processes IAB category results from transcript and provides:
    - Topic labels with relevance scores
    - Category hierarchy parsing
    - Top topics summary

    IAB categories follow a hierarchical format like:
    "Technology&Computing>ArtificialIntelligence"

    Example output:
        {
            "items": [
                {
                    "text": "We are using machine learning...",
                    "labels": [
                        {
                            "label": "Technology&Computing>ArtificialIntelligence",
                            "relevance": 0.95,
                            "category": "Technology&Computing",
                            "subcategory": "ArtificialIntelligence",
                        }
                    ]
                }
            ],
            "count": 1,
            "summary": {
                "Technology&Computing>ArtificialIntelligence": 0.95
            },
            "top_topics": ["ArtificialIntelligence"],
        }
    """

    @property
    def extractor_name(self) -> str:
        """Return the name of this extractor."""
        return "topic"

    def _get_data_key(self) -> str:
        """Return the key for topic data in transcript."""
        return "iab_categories"

    def _do_extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract topic/IAB category data.

        :param data: Raw IAB category data
        :return: Processed topic data
        """
        # Handle different data structures from provider
        results = data.get("results", [])
        summary = data.get("summary", {})

        items = []
        for result in results:
            text = result.get("text", "")
            labels = []

            for label_data in result.get("labels", []):
                label = label_data.get("label", "")
                relevance = label_data.get("relevance", 0.0)
                category, subcategory = self._parse_iab_label(label)

                labels.append({
                    "label": label,
                    "relevance": relevance,
                    "category": category,
                    "subcategory": subcategory,
                })

            items.append({
                "text": text,
                "labels": labels,
            })

        # Process summary for top topics
        top_topics = self._get_top_topics(summary)

        logger.debug(
            "Topic extraction: %d segments, %d unique topics",
            len(items),
            len(summary),
        )

        return {
            "items": items,
            "count": len(items),
            "summary": summary,
            "top_topics": top_topics,
            "unique_categories": self._get_unique_categories(summary),
        }

    def _parse_iab_label(self, label: str) -> Tuple[str, str]:
        """
        Parse IAB label into category and subcategory.

        :param label: Full IAB label (e.g., "Technology>AI")
        :return: Tuple of (category, subcategory)
        """
        parts = label.split(">")
        category = parts[0] if parts else ""
        subcategory = parts[-1] if len(parts) > 1 else ""
        return category, subcategory

    def _get_top_topics(
        self,
        summary: Dict[str, float],
        limit: int = 5,
    ) -> List[str]:
        """
        Get top topics by relevance.

        :param summary: Topic summary with relevance scores
        :param limit: Maximum number of topics to return
        :return: List of top topic labels
        """
        sorted_topics = sorted(
            summary.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [topic for topic, _ in sorted_topics[:limit]]

    def _get_unique_categories(
        self,
        summary: Dict[str, float],
    ) -> List[str]:
        """
        Get unique top-level categories.

        :param summary: Topic summary
        :return: List of unique category names
        """
        categories = set()
        for label in summary.keys():
            category, _ = self._parse_iab_label(label)
            if category:
                categories.add(category)
        return sorted(categories)

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty topic result."""
        return {
            "items": [],
            "count": 0,
            "summary": {},
            "top_topics": [],
            "unique_categories": [],
        }

    def get_top_topics(
        self,
        transcript_data: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get top topics with their relevance scores.

        :param transcript_data: Transcript data
        :param limit: Maximum number of topics
        :return: List of top topics with details
        """
        result = self.extract(transcript_data)
        summary = result["summary"]

        sorted_topics = sorted(
            summary.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        return [
            {
                "label": label,
                "relevance": relevance,
                "category": self._parse_iab_label(label)[0],
                "subcategory": self._parse_iab_label(label)[1],
            }
            for label, relevance in sorted_topics
        ]

    def get_topics_by_category(
        self,
        transcript_data: Dict[str, Any],
        category: str,
    ) -> List[Dict[str, Any]]:
        """
        Get topics filtered by top-level category.

        :param transcript_data: Transcript data
        :param category: Category to filter by
        :return: List of matching topics
        """
        result = self.extract(transcript_data)
        summary = result["summary"]

        matching = []
        for label, relevance in summary.items():
            cat, subcat = self._parse_iab_label(label)
            if cat.lower() == category.lower():
                matching.append({
                    "label": label,
                    "relevance": relevance,
                    "subcategory": subcat,
                })

        return sorted(matching, key=lambda x: x["relevance"], reverse=True)
