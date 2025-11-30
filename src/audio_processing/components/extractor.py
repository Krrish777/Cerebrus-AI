"""
Data extractor Haystack component.

Provides a Haystack-compatible component for extracting data from transcripts.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack import component

from src.audio_processing.extractors.registry import ExtractorRegistry
from src.audio_processing.extractors.registry import get_registry
from src.core.logging import get_logger

logger = get_logger(__name__)


@component
class DataExtractorComponent:
    """
    Haystack component for data extraction from transcripts.
    
    Extracts structured data (sentiment, entities, topics, etc.)
    from transcript data using configured extractors.
    
    Inputs:
        transcripts: List of transcript dictionaries
        
    Outputs:
        transcripts: Original transcripts
        extracted_data: List of extracted data dictionaries
    """
    
    def __init__(
        self,
        extractors: Optional[List[str]] = None,
        registry: Optional[ExtractorRegistry] = None,
    ) -> None:
        """
        Initialize the data extractor component.
        
        Args:
            extractors: List of extractor names to use.
                       If None, uses all available extractors.
            registry: Optional custom extractor registry
        """
        self._registry = registry or get_registry()
        self._extractor_names = extractors or self._registry.available()
    
    @component.output_types(
        transcripts=List[Dict[str, Any]],
        extracted_data=List[Dict[str, Dict[str, Any]]],
    )
    def run(
        self,
        transcripts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extract data from transcripts.
        
        Args:
            transcripts: List of transcript dictionaries
            
        Returns:
            Dictionary with:
                - transcripts: Original transcripts (passed through)
                - extracted_data: List of extraction results
        """
        all_extracted = []
        
        for transcript in transcripts:
            # Skip error transcripts
            if transcript.get("status") == "error":
                all_extracted.append({})
                continue
            
            extracted = self._extract_from_transcript(transcript)
            all_extracted.append(extracted)
            
            logger.info(
                "Extracted %d features from transcript %s",
                len(extracted),
                transcript.get("id", "unknown"),
            )
        
        return {
            "transcripts": transcripts,
            "extracted_data": all_extracted,
        }
    
    def _extract_from_transcript(
        self,
        transcript: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract all configured features from a single transcript.
        
        Args:
            transcript: Transcript dictionary
            
        Returns:
            Dictionary mapping extractor names to extracted data
        """
        extracted = {}
        
        for extractor_name in self._extractor_names:
            try:
                extractor = self._registry.get(extractor_name)
                
                if extractor.is_available(transcript):
                    data = extractor.extract(transcript)
                    extracted[extractor_name] = data
                    
                    logger.debug(
                        "Extracted %s from transcript",
                        extractor_name,
                    )
                else:
                    logger.debug(
                        "Extractor %s not available for transcript",
                        extractor_name,
                    )
                    
            except Exception as e:
                logger.warning(
                    "Failed to extract %s: %s",
                    extractor_name,
                    str(e),
                )
        
        return extracted
