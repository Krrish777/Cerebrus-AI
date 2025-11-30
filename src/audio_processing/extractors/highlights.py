"""
Highlights extractor implementation.

Extracts auto-highlights/key phrases from transcripts.
"""

from typing import Any, Dict, List

from src.audio_processing.extractors.base import BaseExtractor
from src.core.logging import get_logger

logger = get_logger(__name__)


class HighlightsExtractor(BaseExtractor):
    """
    Extractor for auto-highlights/key phrases.

    Processes auto-highlights results from transcript and provides:
    - Key phrases with timestamps
    - Ranked highlights by importance
    - Occurrence counts

    Example output:
        {
            "items": [
                {
                    "text": "machine learning",
                    "count": 5,
                    "rank": 0.95,
                    "timestamps": [
                        {"start": 1000, "end": 1500},
                        {"start": 5000, "end": 5500},
                    ]
                }
            ],
            "count": 1,
            "top_highlights": ["machine learning"],
        }
    """

    @property
    def extractor_name(self) -> str:
        """Return the name of this extractor."""
        return "highlights"

    def _get_data_key(self) -> str:
        """Return the key for highlights data in transcript."""
        return "auto_highlights"

    def _do_extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract highlights data.

        :param data: Raw highlights data
        :return: Processed highlights data
        """
        results = data.get("results", [])

        items = []
        for result in results:
            text = result.get("text", "")
            count = result.get("count", 0)
            rank = result.get("rank", 0.0)
            timestamps = result.get("timestamps", [])

            processed_timestamps = [
                {
                    "start": ts.get("start", 0),
                    "end": ts.get("end", 0),
                    "start_seconds": ts.get("start", 0) / 1000.0,
                    "end_seconds": ts.get("end", 0) / 1000.0,
                }
                for ts in timestamps
            ]

            items.append({
                "text": text,
                "count": count,
                "rank": rank,
                "timestamps": processed_timestamps,
                "first_occurrence": processed_timestamps[0] if processed_timestamps else None,
            })

        # Sort by rank (descending)
        items.sort(key=lambda x: x["rank"], reverse=True)

        # Get top highlights
        top_highlights = [item["text"] for item in items[:10]]

        logger.debug(
            "Highlights extraction: %d key phrases",
            len(items),
        )

        return {
            "items": items,
            "count": len(items),
            "top_highlights": top_highlights,
            "total_occurrences": sum(item["count"] for item in items),
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty highlights result."""
        return {
            "items": [],
            "count": 0,
            "top_highlights": [],
            "total_occurrences": 0,
        }

    def get_top_highlights(
        self,
        transcript_data: Dict[str, Any],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top highlights by rank.

        :param transcript_data: Transcript data
        :param limit: Maximum number of highlights
        :return: List of top highlights
        """
        result = self.extract(transcript_data)
        return result["items"][:limit]

    def get_most_frequent(
        self,
        transcript_data: Dict[str, Any],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get most frequently occurring highlights.

        :param transcript_data: Transcript data
        :param limit: Maximum number of highlights
        :return: List of most frequent highlights
        """
        result = self.extract(transcript_data)
        sorted_items = sorted(
            result["items"],
            key=lambda x: x["count"],
            reverse=True,
        )
        return sorted_items[:limit]

    def get_highlights_at_time(
        self,
        transcript_data: Dict[str, Any],
        time_ms: int,
    ) -> List[Dict[str, Any]]:
        """
        Get highlights that occur at a specific time.

        :param transcript_data: Transcript data
        :param time_ms: Time in milliseconds
        :return: List of highlights at that time
        """
        result = self.extract(transcript_data)

        matching = []
        for item in result["items"]:
            for ts in item["timestamps"]:
                if ts["start"] <= time_ms <= ts["end"]:
                    matching.append(item)
                    break

        return matching

    def get_highlight_timeline(
        self,
        transcript_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Get all highlights in chronological order.

        :param transcript_data: Transcript data
        :return: List of highlights ordered by first occurrence
        """
        result = self.extract(transcript_data)

        timeline = []
        for item in result["items"]:
            if item["timestamps"]:
                timeline.append({
                    "text": item["text"],
                    "start": item["timestamps"][0]["start"],
                    "end": item["timestamps"][0]["end"],
                    "rank": item["rank"],
                })

        return sorted(timeline, key=lambda x: x["start"])

    def search_highlights(
        self,
        transcript_data: Dict[str, Any],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Search for highlights containing a query string.

        :param transcript_data: Transcript data
        :param query: Search query
        :return: Matching highlights
        """
        result = self.extract(transcript_data)
        query_lower = query.lower()

        return [
            item for item in result["items"]
            if query_lower in item["text"].lower()
        ]
