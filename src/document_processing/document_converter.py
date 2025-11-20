"""
Document Converter

Orchestrates document conversion using Haystack components.
Follows AGENTS.md principles: single responsibility, loose coupling, extensibility.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument, MarkdownToDocument
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.document_processing.pipeline_config import PipelineConfig

logger = get_logger(__name__)


class DocumentConverter:
    """
    Converts files to Haystack Document objects using appropriate converters.
    
    Orchestrates the conversion process with proper error handling and
    performance monitoring. Uses Haystack components for actual conversion.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize document converter with configuration.

        :param config: Pipeline configuration
        """
        self.config = config
        self._pipeline = None
        logger.debug("Document converter initialized")

    @property
    def pipeline(self) -> Pipeline:
        """Lazy-loaded conversion pipeline."""
        if self._pipeline is None:
            self._pipeline = self._build_conversion_pipeline()
        return self._pipeline

    def convert_files(self, file_type: str, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Convert files to Document objects.

        :param file_type: Type of files being converted
        :param file_paths: List of file paths to convert
        :return: Conversion results with documents and errors
        """
        start_time = time.time()
        logger.info("Converting %d %s files", len(file_paths), file_type)

        if not file_paths:
            logger.warning("No files provided for conversion")
            return {"documents": [], "errors": []}

        try:
            # Convert paths to strings for Haystack compatibility
            file_sources = [str(path) for path in file_paths]
            
            # Run conversion pipeline
            result = self.pipeline.run({
                "router": {"sources": file_sources}
            })

            # Extract documents from pipeline result
            documents = self._extract_documents_from_result(result)
            
            # Enhance documents with source information
            enhanced_docs = []
            for doc in documents:
                enhanced_doc = self._enhance_document_metadata(doc, file_type)
                enhanced_docs.append(enhanced_doc)

            conversion_time = time.time() - start_time
            logger.info(
                "Converted %d files to %d documents in %.2fs",
                len(file_paths), len(enhanced_docs), conversion_time
            )

            return {
                "documents": enhanced_docs,
                "errors": [],
                "stats": {
                    "conversion_time": conversion_time,
                    "files_converted": len(file_paths),
                    "documents_created": len(enhanced_docs)
                }
            }

        except Exception as e:
            error_msg = f"Document conversion failed for {file_type} files: {str(e)}"
            logger.error(error_msg)
            
            return {
                "documents": [],
                "errors": [error_msg],
                "stats": {
                    "conversion_time": time.time() - start_time,
                    "files_converted": 0,
                    "documents_created": 0
                }
            }

    def _build_conversion_pipeline(self) -> Pipeline:
        """
        Build the document conversion pipeline.

        :return: Configured Haystack Pipeline
        """
        logger.debug("Building document conversion pipeline")
        
        pipeline = Pipeline()

        # Add file type router
        router = FileTypeRouter(
            mime_types=self.config.file_types.supported_mime_types
        )
        pipeline.add_component("router", router)

        # Add converters based on configuration
        if self.config.processing.enable_pdf_processing:
            pdf_converter = PyPDFToDocument()
            pipeline.add_component("pdf_converter", pdf_converter)

        if self.config.processing.enable_text_processing:
            text_converter = TextFileToDocument()
            pipeline.add_component("text_converter", text_converter)

        # Add markdown converter if enabled
        if self.config.processing.enable_markdown_processing:
            markdown_converter = MarkdownToDocument()
            pipeline.add_component("markdown_converter", markdown_converter)
            logger.debug("Markdown converter added to pipeline")

        # Add document joiner
        joiner = DocumentJoiner()
        pipeline.add_component("joiner", joiner)

        # Connect components
        self._connect_pipeline_components(pipeline)

        logger.debug("Document conversion pipeline built successfully")
        return pipeline

    def _connect_pipeline_components(self, pipeline: Pipeline) -> None:
        """
        Connect pipeline components.

        :param pipeline: Pipeline to configure
        """
        # Get available output sockets from router
        router_outputs = list(pipeline.graph.nodes["router"]["output_sockets"].keys())
        
        # Connect router to appropriate converters
        if "pdf_converter" in pipeline.graph.nodes and "application/pdf" in router_outputs:
            pipeline.connect("router.application/pdf", "pdf_converter")
            pipeline.connect("pdf_converter", "joiner")

        if "text_converter" in pipeline.graph.nodes and "text/plain" in router_outputs:
            pipeline.connect("router.text/plain", "text_converter")
            pipeline.connect("text_converter", "joiner")

        if "markdown_converter" in pipeline.graph.nodes and "text/markdown" in router_outputs:
            pipeline.connect("router.text/markdown", "markdown_converter")
            pipeline.connect("markdown_converter", "joiner")
        elif "text_converter" in pipeline.graph.nodes and "text/markdown" in router_outputs:
            # Fallback: route markdown files to text converter if markdown processing is disabled
            pipeline.connect("router.text/markdown", "text_converter")

        # Handle unclassified files - route to text converter if available
        if "text_converter" in pipeline.graph.nodes and "unclassified" in router_outputs:
            pipeline.connect("router.unclassified", "text_converter")

        logger.debug("Pipeline components connected")

    def _extract_documents_from_result(self, pipeline_result: Dict[str, Any]) -> List[Document]:
        """
        Extract documents from pipeline execution result.

        :param pipeline_result: Result from pipeline.run()
        :return: List of Document objects
        """
        # Try to get documents from joiner first
        if "joiner" in pipeline_result:
            return pipeline_result["joiner"].get("documents", [])

        # Fallback: collect documents from individual converters
        documents = []
        
        for component_name, component_result in pipeline_result.items():
            if isinstance(component_result, dict) and "documents" in component_result:
                documents.extend(component_result["documents"])

        return documents

    def _enhance_document_metadata(self, document: Document, file_type: str) -> Document:
        """
        Enhance document metadata with conversion information.

        :param document: Original document
        :param file_type: Type of source file
        :return: Enhanced document with additional metadata
        """
        # Create enhanced metadata
        enhanced_meta = document.meta.copy()
        
        # Add conversion metadata
        enhanced_meta.update({
            "source_type": file_type.lower(),
            "converter": self._get_converter_name(file_type),
            "conversion_timestamp": time.time(),
            "pipeline_version": "1.0"
        })

        # Ensure source file information is available
        if "file_path" not in enhanced_meta and hasattr(document, 'meta'):
            # Try to extract from existing metadata
            source_file = enhanced_meta.get("name") or enhanced_meta.get("source")
            if source_file:
                enhanced_meta["file_path"] = source_file

        # Create new document with enhanced metadata
        return Document(
            id=document.id,
            content=document.content,
            meta=enhanced_meta,
            embedding=document.embedding
        )

    def _get_converter_name(self, file_type: str) -> str:
        """
        Get the converter name used for a file type.

        :param file_type: File type
        :return: Converter component name
        """
        converter_mapping = {
            "PDF": "PyPDFToDocument",
            "Text": "TextFileToDocument",
            "Markdown": "MarkdownToDocument"
        }
        
        return converter_mapping.get(file_type, "TextFileToDocument")

    def get_supported_types(self) -> List[str]:
        """
        Get list of supported file types.

        :return: List of supported file type names
        """
        supported_types = []
        
        if self.config.processing.enable_pdf_processing:
            supported_types.append("PDF")
            
        if self.config.processing.enable_text_processing:
            supported_types.append("Text")
            
        if self.config.processing.enable_markdown_processing:
            supported_types.append("Markdown")
        
        return supported_types

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the conversion pipeline.

        :return: Pipeline information
        """
        return {
            "components": list(self.pipeline.graph.nodes.keys()),
            "connections": len(self.pipeline.graph.edges),
            "supported_types": self.get_supported_types(),
            "configuration": {
                "pdf_enabled": self.config.processing.enable_pdf_processing,
                "text_enabled": self.config.processing.enable_text_processing,
                "markdown_enabled": self.config.processing.enable_markdown_processing,
                "markdown_fallback": self.config.processing.enable_markdown_fallback
            }
        }