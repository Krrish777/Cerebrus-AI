"""
Entity extractor implementation.

Extracts named entities from transcripts.
"""

from collections import defaultdict
from typing import Any, Dict, List

from src.audio_processing.extractors.base import BaseExtractor
from src.core.logging import get_logger

logger = get_logger(__name__)


class EntityExtractor(BaseExtractor):
    """
    Extractor for named entity recognition data.

    Processes entity detection results from transcript and provides:
    - Individual entities with positions
    - Entity type grouping
    - Entity frequency counts

    Supported entity types:
    - PERSON: People's names
    - ORGANIZATION: Company/organization names
    - LOCATION: Places and addresses
    - DATE: Dates and times
    - MONEY: Monetary values
    - PHONE_NUMBER: Phone numbers
    - EMAIL: Email addresses
    - URL: Web URLs
    - And more depending on provider

    Example output:
        {
            "items": [
                {
                    "text": "John Smith",
                    "entity_type": "PERSON",
                    "start": 1000,
                    "end": 1500,
                }
            ],
            "count": 1,
            "by_type": {"PERSON": ["John Smith"]},
            "unique_entities": 1,
        }
    """

    @property
    def extractor_name(self) -> str:
        """Return the name of this extractor."""
        return "entity"

    def _get_data_key(self) -> str:
        """Return the key for entity data in transcript."""
        return "entities"

    def _do_extract(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract entity data.

        :param data: Raw entity detection data
        :return: Processed entity data
        """
        items = []
        by_type: Dict[str, List[str]] = defaultdict(list)
        unique_entities: set = set()

        for item in data:
            entity_type = self._normalize_entity_type(
                item.get("entity_type", "UNKNOWN")
            )
            text = item.get("text", "")

            processed_item = {
                "text": text,
                "entity_type": entity_type,
                "start": item.get("start", 0),
                "end": item.get("end", 0),
            }

            items.append(processed_item)

            # Track unique entities by type
            entity_key = f"{entity_type}:{text.lower()}"
            if entity_key not in unique_entities:
                unique_entities.add(entity_key)
                by_type[entity_type].append(text)

        # Convert defaultdict to regular dict
        by_type_dict = dict(by_type)

        logger.debug(
            "Entity extraction: %d items, %d unique across %d types",
            len(items),
            len(unique_entities),
            len(by_type_dict),
        )

        return {
            "items": items,
            "count": len(items),
            "by_type": by_type_dict,
            "unique_entities": len(unique_entities),
            "entity_types": list(by_type_dict.keys()),
        }

    def _normalize_entity_type(self, entity_type: str) -> str:
        """
        Normalize entity type to standard format.

        :param entity_type: Raw entity type string
        :return: Normalized entity type
        """
        # Convert to uppercase and remove common prefixes
        normalized = str(entity_type).upper().strip()

        # Handle common variations
        type_mapping = {
            "PERSON_NAME": "PERSON",
            "PERSON": "PERSON",
            "ORGANIZATION_NAME": "ORGANIZATION",
            "ORG": "ORGANIZATION",
            "LOCATION": "LOCATION",
            "LOC": "LOCATION",
            "GPE": "LOCATION",
            "DATE": "DATE",
            "TIME": "DATE",
            "DATETIME": "DATE",
            "MONEY": "MONEY",
            "MONETARY_VALUE": "MONEY",
            "PHONE_NUMBER": "PHONE_NUMBER",
            "PHONE": "PHONE_NUMBER",
            "EMAIL_ADDRESS": "EMAIL",
            "EMAIL": "EMAIL",
            "URL": "URL",
            "WEBSITE": "URL",
            "CREDIT_CARD": "CREDIT_CARD",
            "CREDIT_CARD_NUMBER": "CREDIT_CARD",
            "SSN": "SSN",
            "SOCIAL_SECURITY_NUMBER": "SSN",
        }

        return type_mapping.get(normalized, normalized)

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty entity result."""
        return {
            "items": [],
            "count": 0,
            "by_type": {},
            "unique_entities": 0,
            "entity_types": [],
        }

    def get_entities_by_type(
        self,
        transcript_data: Dict[str, Any],
        entity_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Get entities filtered by type.

        :param transcript_data: Transcript data
        :param entity_type: Entity type to filter by
        :return: Filtered entity items
        """
        result = self.extract(transcript_data)
        normalized_type = self._normalize_entity_type(entity_type)

        return [
            item for item in result["items"]
            if item["entity_type"] == normalized_type
        ]

    def get_unique_entities(
        self,
        transcript_data: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """
        Get unique entities grouped by type.

        :param transcript_data: Transcript data
        :return: Dictionary of entity type to unique entity texts
        """
        result = self.extract(transcript_data)
        return result["by_type"]

    def get_pii_entities(
        self,
        transcript_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Get potentially sensitive PII entities.

        :param transcript_data: Transcript data
        :return: List of PII entity items
        """
        pii_types = {
            "PERSON",
            "EMAIL",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "SSN",
            "DATE_OF_BIRTH",
            "BANK_ACCOUNT",
            "DRIVER_LICENSE",
            "PASSPORT_NUMBER",
        }

        result = self.extract(transcript_data)

        return [
            item for item in result["items"]
            if item["entity_type"] in pii_types
        ]
