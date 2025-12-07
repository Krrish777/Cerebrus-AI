"""
Input validation utilities.
"""

from typing import Any, Dict, Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def validate_query(query: str) -> str:
    """
    Validate and sanitize query string.
    
    Args:
        query: Query string to validate
        
    Returns:
        Sanitized query string
        
    Raises:
        ValidationError: If query is invalid
    """
    if not query:
        raise ValidationError("Query cannot be empty")
    
    if not isinstance(query, str):
        raise ValidationError(f"Query must be string, got {type(query)}")
    
    # Strip whitespace
    query = query.strip()
    
    if not query:
        raise ValidationError("Query cannot be empty after stripping whitespace")
    
    # Check length
    if len(query) > 10000:
        logger.warning(f"Query exceeds recommended length: {len(query)}")
        query = query[:10000]
    
    return query


def validate_top_k(top_k: Optional[int], min_val: int = 1, max_val: int = 100) -> Optional[int]:
    """
    Validate top_k parameter.
    
    Args:
        top_k: Top-k value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated top_k or None
        
    Raises:
        ValidationError: If top_k is invalid
    """
    if top_k is None:
        return None
    
    if not isinstance(top_k, int):
        raise ValidationError(f"top_k must be integer, got {type(top_k)}")
    
    if top_k < min_val:
        raise ValidationError(f"top_k must be >= {min_val}, got {top_k}")
    
    if top_k > max_val:
        logger.warning(f"top_k {top_k} exceeds maximum {max_val}, capping")
        return max_val
    
    return top_k


def validate_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Validate filters dictionary.
    
    Args:
        filters: Filters to validate
        
    Returns:
        Validated filters or None
        
    Raises:
        ValidationError: If filters are invalid
    """
    if filters is None:
        return None
    
    if not isinstance(filters, dict):
        raise ValidationError(f"Filters must be dictionary, got {type(filters)}")
    
    # Check for empty dict
    if not filters:
        return None
    
    return filters


def validate_score_threshold(threshold: Optional[float]) -> Optional[float]:
    """
    Validate score threshold.
    
    Args:
        threshold: Threshold to validate
        
    Returns:
        Validated threshold or None
        
    Raises:
        ValidationError: If threshold is invalid
    """
    if threshold is None:
        return None
    
    if not isinstance(threshold, (int, float)):
        raise ValidationError(f"Threshold must be numeric, got {type(threshold)}")
    
    threshold = float(threshold)
    
    if threshold < 0.0 or threshold > 1.0:
        raise ValidationError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
    
    return threshold
