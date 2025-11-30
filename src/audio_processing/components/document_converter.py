"""
Document converter Haystack component.

Provides a Haystack-compatible component for converting transcripts to Documents.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack import component
from haystack.dataclasses import Document

from src.audio_processing.chunking.base import Chunk
from src.audio_processing.document.builder import TranscriptDocumentBuilder
from src.core.logging import get_logger

logger = get_logger(__name__)


@component
class DocumentConverterComponent:
    """
    Haystack component for converting transcripts to Documents.
    
    Converts transcript data and extracted features into
    Haystack Document objects.
    
    Inputs:
        transcripts: List of transcript dictionaries
        extracted_data: List of extracted data dictionaries
        source_names: List of source names
        chunks: Optional list of chunk lists
        
    Outputs:
        documents: List of Haystack Documents
    """
    
    def __init__(
        self,
        use_chunks: bool = True,
        builder: Optional[TranscriptDocumentBuilder] = None,
    ) -> None:
        """
        Initialize the document converter component.
        
        Args:
            use_chunks: Whether to create documents from chunks
            builder: Optional custom document builder
        """
        self._use_chunks = use_chunks
        self._builder = builder or TranscriptDocumentBuilder()
    
    @component.output_types(documents=List[Document])
    def run(
        self,
        transcripts: List[Dict[str, Any]],
        extracted_data: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        source_names: Optional[List[str]] = None,
        chunks: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
        """
        Convert transcripts to documents.
        
        Args:
            transcripts: List of transcript dictionaries
            extracted_data: Optional list of extracted data per transcript
            source_names: Optional list of source names per transcript
            chunks: Optional list of chunk lists per transcript
            
        Returns:
            Dictionary with 'documents' key containing Document list
        """
        all_documents = []
        
        # Ensure extracted_data has same length as transcripts
        if extracted_data is None:
            extracted_data = [{}] * len(transcripts)
        
        # Ensure source_names has same length
        if source_names is None:
            source_names = [
                t.get("source", f"audio_{i}")
                for i, t in enumerate(transcripts)
            ]
        
        # Ensure chunks has same length if provided
        if chunks is None and self._use_chunks:
            chunks = [None] * len(transcripts)
        
        for i, transcript in enumerate(transcripts):
            # Skip error transcripts
            if transcript.get("status") == "error":
                continue
            
            source_name = source_names[i]
            extracted = extracted_data[i]
            transcript_chunks = chunks[i] if chunks else None
            
            if transcript_chunks and self._use_chunks:
                # Convert serialized chunks back to Chunk objects
                chunk_objects = [
                    self._deserialize_chunk(c) for c in transcript_chunks
                ]
                
                docs = self._builder.build_from_chunks(
                    chunks=chunk_objects,
                    transcript_data=transcript,
                    extracted_data=extracted,
                    source_name=source_name,
                )
            else:
                docs = self._builder.build(
                    transcript_data=transcript,
                    extracted_data=extracted,
                    source_name=source_name,
                )
            
            all_documents.extend(docs)
            
            logger.info(
                "Created %d documents from transcript %s",
                len(docs),
                transcript.get("id", "unknown"),
            )
        
        return {"documents": all_documents}
    
    def _deserialize_chunk(self, chunk_dict: Dict[str, Any]) -> Chunk:
        """
        Convert a serialized chunk dictionary to a Chunk object.
        
        Args:
            chunk_dict: Serialized chunk dictionary
            
        Returns:
            Chunk object
        """
        return Chunk(
            text=chunk_dict.get("text", ""),
            start_time=chunk_dict.get("start_time", 0),
            end_time=chunk_dict.get("end_time", 0),
            speaker=chunk_dict.get("speaker"),
            metadata=chunk_dict.get("metadata", {}),
        )
