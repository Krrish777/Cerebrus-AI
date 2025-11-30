"""
Content safety extractor implementation.

Extracts content safety/moderation data from transcripts.
"""

from collections import defaultdict
from typing import Any, Dict, List

from src.audio_processing.extractors.base import BaseExtractor
from src.core.logging import get_logger

logger = get_logger(__name__)


class ContentSafetyExtractor(BaseExtractor):
    """
    Extractor for content safety/moderation data.

    Processes content safety results from transcript and provides:
    - Flagged content with severity levels
    - Safety label distribution
    - Overall safety assessment

    Safety labels include:
    - profanity: Profane language
    - hate_speech: Hateful content
    - violence: Violent content
    - sexual_content: Sexual content
    - drug_use: Drug-related content
    - gambling: Gambling references
    - weapons: Weapon references

    Example output:
        {
            "items": [
                {
                    "text": "Some flagged content...",
                    "labels": [
                        {
                            "label": "profanity",
                            "confidence": 0.95,
                            "severity": 0.5,
                        }
                    ],
                    "start": 1000,
                    "end": 2000,
                }
            ],
            "count": 1,
            "summary": {"profanity": 0.5},
            "is_safe": False,
            "severity_score": 0.5,
        }
    """

    # Threshold for considering content as unsafe
    SAFETY_THRESHOLD = 0.5

    @property
    def extractor_name(self) -> str:
        """Return the name of this extractor."""
        return "content_safety"

    def _get_data_key(self) -> str:
        """Return the key for content safety data in transcript."""
        return "content_safety"

    def _do_extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content safety data.

        :param data: Raw content safety data
        :return: Processed safety data
        """
        results = data.get("results", [])
        summary = data.get("summary", {})

        items = []
        label_counts: Dict[str, int] = defaultdict(int)
        max_severity = 0.0

        for result in results:
            text = result.get("text", "")
            labels = []

            for label_data in result.get("labels", []):
                label = label_data.get("label", "")
                confidence = label_data.get("confidence", 0.0)
                severity = label_data.get("severity", 0.0)

                labels.append({
                    "label": label,
                    "confidence": confidence,
                    "severity": severity,
                })

                label_counts[label] += 1
                max_severity = max(max_severity, severity)

            if labels:
                items.append({
                    "text": text,
                    "labels": labels,
                    "start": result.get("start", 0),
                    "end": result.get("end", 0),
                })

        # Determine overall safety
        is_safe = max_severity < self.SAFETY_THRESHOLD and len(items) == 0

        logger.debug(
            "Content safety extraction: %d flags, max_severity=%.2f, is_safe=%s",
            len(items),
            max_severity,
            is_safe,
        )

        return {
            "items": items,
            "count": len(items),
            "summary": dict(summary),
            "label_counts": dict(label_counts),
            "is_safe": is_safe,
            "severity_score": round(max_severity, 4),
            "flagged_labels": list(label_counts.keys()),
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty content safety result."""
        return {
            "items": [],
            "count": 0,
            "summary": {},
            "label_counts": {},
            "is_safe": True,
            "severity_score": 0.0,
            "flagged_labels": [],
        }

    def is_safe(self, transcript_data: Dict[str, Any]) -> bool:
        """
        Quick check if content is safe.

        :param transcript_data: Transcript data
        :return: True if content is considered safe
        """
        result = self.extract(transcript_data)
        return result["is_safe"]

    def get_flags_by_label(
        self,
        transcript_data: Dict[str, Any],
        label: str,
    ) -> List[Dict[str, Any]]:
        """
        Get flagged content by specific label.

        :param transcript_data: Transcript data
        :param label: Safety label to filter by
        :return: List of flagged items with that label
        """
        result = self.extract(transcript_data)
        label_lower = label.lower()

        matching = []
        for item in result["items"]:
            matching_labels = [
                lbl for lbl in item["labels"]
                if lbl["label"].lower() == label_lower
            ]
            if matching_labels:
                matching.append({
                    "text": item["text"],
                    "labels": matching_labels,
                    "start": item["start"],
                    "end": item["end"],
                })

        return matching

    def get_high_severity_flags(
        self,
        transcript_data: Dict[str, Any],
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Get content flags above a severity threshold.

        :param transcript_data: Transcript data
        :param threshold: Minimum severity (0-1)
        :return: List of high-severity flagged items
        """
        result = self.extract(transcript_data)

        high_severity = []
        for item in result["items"]:
            severe_labels = [
                lbl for lbl in item["labels"]
                if lbl["severity"] >= threshold
            ]
            if severe_labels:
                high_severity.append({
                    "text": item["text"],
                    "labels": severe_labels,
                    "start": item["start"],
                    "end": item["end"],
                })

        return high_severity

    def get_safety_summary(
        self,
        transcript_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get a summary of content safety.

        :param transcript_data: Transcript data
        :return: Safety summary
        """
        result = self.extract(transcript_data)

        return {
            "is_safe": result["is_safe"],
            "severity_score": result["severity_score"],
            "flagged_labels": result["flagged_labels"],
            "flag_count": result["count"],
        }
