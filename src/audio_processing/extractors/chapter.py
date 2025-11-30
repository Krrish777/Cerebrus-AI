"""
Chapter extractor implementation.

Extracts auto-generated chapters from transcripts.
"""

from typing import Any, Dict, List, Optional

from src.audio_processing.extractors.base import BaseExtractor
from src.core.logging import get_logger

logger = get_logger(__name__)


class ChapterExtractor(BaseExtractor):
    """
    Extractor for auto-generated chapters.

    Processes chapter data from transcript and provides:
    - Chapter list with headlines, summaries, and gists
    - Time-based chapter navigation
    - Chapter duration calculations

    Example output:
        {
            "items": [
                {
                    "headline": "Introduction",
                    "summary": "The speaker introduces the topic...",
                    "gist": "Topic introduction",
                    "start": 0,
                    "end": 60000,
                    "duration": 60.0,
                    "index": 0,
                }
            ],
            "count": 1,
            "total_duration": 60.0,
        }
    """

    @property
    def extractor_name(self) -> str:
        """Return the name of this extractor."""
        return "chapter"

    def _get_data_key(self) -> str:
        """Return the key for chapter data in transcript."""
        return "chapters"

    def _do_extract(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract chapter data.

        :param data: Raw chapter data
        :return: Processed chapter data
        """
        items = []
        total_duration = 0.0

        for index, item in enumerate(data):
            start_ms = item.get("start", 0)
            end_ms = item.get("end", 0)
            duration = (end_ms - start_ms) / 1000.0  # Convert to seconds

            processed_item = {
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "gist": item.get("gist", ""),
                "start": start_ms,
                "end": end_ms,
                "start_seconds": start_ms / 1000.0,
                "end_seconds": end_ms / 1000.0,
                "duration": round(duration, 2),
                "index": index,
            }

            items.append(processed_item)
            total_duration += duration

        logger.debug(
            "Chapter extraction: %d chapters, total duration %.2fs",
            len(items),
            total_duration,
        )

        return {
            "items": items,
            "count": len(items),
            "total_duration": round(total_duration, 2),
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty chapter result."""
        return {
            "items": [],
            "count": 0,
            "total_duration": 0.0,
        }

    def get_chapter_at_time(
        self,
        transcript_data: Dict[str, Any],
        time_ms: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the chapter at a specific time.

        :param transcript_data: Transcript data
        :param time_ms: Time in milliseconds
        :return: Chapter at the given time or None
        """
        result = self.extract(transcript_data)

        for chapter in result["items"]:
            if chapter["start"] <= time_ms < chapter["end"]:
                return chapter

        return None

    def get_chapter_by_index(
        self,
        transcript_data: Dict[str, Any],
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a chapter by its index.

        :param transcript_data: Transcript data
        :param index: Chapter index (0-based)
        :return: Chapter at the given index or None
        """
        result = self.extract(transcript_data)

        if 0 <= index < len(result["items"]):
            return result["items"][index]

        return None

    def get_headlines(
        self,
        transcript_data: Dict[str, Any],
    ) -> List[str]:
        """
        Get all chapter headlines.

        :param transcript_data: Transcript data
        :return: List of chapter headlines
        """
        result = self.extract(transcript_data)
        return [chapter["headline"] for chapter in result["items"]]

    def get_table_of_contents(
        self,
        transcript_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Get a table of contents structure.

        :param transcript_data: Transcript data
        :return: List of TOC entries with headline and timing
        """
        result = self.extract(transcript_data)

        return [
            {
                "index": chapter["index"],
                "headline": chapter["headline"],
                "gist": chapter["gist"],
                "start_seconds": chapter["start_seconds"],
                "duration": chapter["duration"],
            }
            for chapter in result["items"]
        ]

    def format_timestamp(self, milliseconds: int) -> str:
        """
        Format milliseconds to MM:SS or HH:MM:SS.

        :param milliseconds: Time in milliseconds
        :return: Formatted time string
        """
        total_seconds = milliseconds // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
