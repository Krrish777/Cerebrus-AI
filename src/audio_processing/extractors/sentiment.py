"""
Sentiment extractor implementation.

Extracts sentiment analysis data from transcripts.
"""

from typing import Any, Dict, List

from src.audio_processing.extractors.base import BaseExtractor
from src.core.logging import get_logger

logger = get_logger(__name__)


class SentimentExtractor(BaseExtractor):
    """
    Extractor for sentiment analysis data.

    Processes sentiment results from transcript and provides:
    - Individual segment sentiments
    - Overall sentiment distribution
    - Aggregated sentiment score

    Example output:
        {
            "items": [
                {
                    "text": "This is great!",
                    "sentiment": "POSITIVE",
                    "confidence": 0.95,
                    "start": 1000,
                    "end": 2500,
                }
            ],
            "count": 1,
            "distribution": {"POSITIVE": 1, "NEGATIVE": 0, "NEUTRAL": 0},
            "dominant_sentiment": "POSITIVE",
            "average_confidence": 0.95,
        }
    """

    @property
    def extractor_name(self) -> str:
        """Return the name of this extractor."""
        return "sentiment"

    def _get_data_key(self) -> str:
        """Return the key for sentiment data in transcript."""
        return "sentiment_analysis"

    def _do_extract(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract sentiment analysis data.

        :param data: Raw sentiment analysis data
        :return: Processed sentiment data
        """
        items = []
        distribution: Dict[str, int] = {
            "POSITIVE": 0,
            "NEGATIVE": 0,
            "NEUTRAL": 0,
        }
        total_confidence = 0.0

        for item in data:
            sentiment = self._normalize_sentiment(item.get("sentiment", ""))
            confidence = item.get("confidence", 0.0)

            processed_item = {
                "text": item.get("text", ""),
                "sentiment": sentiment,
                "confidence": confidence,
                "start": item.get("start", 0),
                "end": item.get("end", 0),
            }

            # Add speaker if available
            if "speaker" in item:
                processed_item["speaker"] = item["speaker"]

            items.append(processed_item)
            distribution[sentiment] = distribution.get(sentiment, 0) + 1
            total_confidence += confidence

        # Calculate aggregates
        count = len(items)
        average_confidence = total_confidence / count if count > 0 else 0.0
        dominant_sentiment = max(distribution, key=lambda k: distribution[k])

        logger.debug(
            "Sentiment extraction: %d items, dominant=%s, avg_conf=%.2f",
            count,
            dominant_sentiment,
            average_confidence,
        )

        return {
            "items": items,
            "count": count,
            "distribution": distribution,
            "dominant_sentiment": dominant_sentiment,
            "average_confidence": round(average_confidence, 4),
        }

    def _normalize_sentiment(self, sentiment: str) -> str:
        """
        Normalize sentiment value to standard format.

        :param sentiment: Raw sentiment string
        :return: Normalized sentiment (POSITIVE, NEGATIVE, NEUTRAL)
        """
        sentiment_upper = str(sentiment).upper()

        if "POSITIVE" in sentiment_upper:
            return "POSITIVE"
        if "NEGATIVE" in sentiment_upper:
            return "NEGATIVE"

        return "NEUTRAL"

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty sentiment result."""
        return {
            "items": [],
            "count": 0,
            "distribution": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
            "dominant_sentiment": "NEUTRAL",
            "average_confidence": 0.0,
        }

    def get_sentiment_summary(
        self,
        transcript_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get a summary of sentiment without full items.

        :param transcript_data: Transcript data
        :return: Sentiment summary
        """
        result = self.extract(transcript_data)

        return {
            "dominant_sentiment": result["dominant_sentiment"],
            "distribution": result["distribution"],
            "average_confidence": result["average_confidence"],
            "count": result["count"],
        }

    def filter_by_sentiment(
        self,
        transcript_data: Dict[str, Any],
        sentiment: str,
    ) -> List[Dict[str, Any]]:
        """
        Filter sentiment items by sentiment type.

        :param transcript_data: Transcript data
        :param sentiment: Sentiment to filter by (POSITIVE, NEGATIVE, NEUTRAL)
        :return: Filtered items
        """
        result = self.extract(transcript_data)
        normalized = self._normalize_sentiment(sentiment)

        return [
            item for item in result["items"]
            if item["sentiment"] == normalized
        ]
