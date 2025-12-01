"""Semantic cache for approved Splunk queries.

Uses sentence embeddings to find semantically similar questions
and return previously approved queries instead of generating new ones.
"""

import re
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from loguru import logger


# Time parameter patterns for extraction and substitution
TIME_PATTERNS = {
    # Pattern to match various time expressions in natural language
    'hours': r'(?:last|past)?\s*(\d+)\s*hours?',
    'days': r'(?:last|past)?\s*(\d+)\s*days?',
    'minutes': r'(?:last|past)?\s*(\d+)\s*minutes?',
    'weeks': r'(?:last|past)?\s*(\d+)\s*weeks?',
    'months': r'(?:last|past)?\s*(\d+)\s*months?',
    # Also match "last day", "last hour" etc (singular without number)
    'last_day': r'(?:last|past)\s+day',
    'last_hour': r'(?:last|past)\s+hour',
    'last_week': r'(?:last|past)\s+week',
    'last_month': r'(?:last|past)\s+month',
}

# Splunk time format pattern in queries
SPLUNK_TIME_PATTERN = r'earliest=-(\d+)([hdwm])'


def extract_time_param(instruction: str) -> Optional[str]:
    """
    Extract time parameter from a natural language instruction.

    Args:
        instruction: The user's instruction

    Returns:
        Splunk-formatted time string (e.g., '-24h', '-7d') or None
    """
    instruction_lower = instruction.lower()

    # Check for specific hour mentions
    hours_match = re.search(TIME_PATTERNS['hours'], instruction_lower)
    if hours_match:
        return f"-{hours_match.group(1)}h"

    # Check for specific day mentions
    days_match = re.search(TIME_PATTERNS['days'], instruction_lower)
    if days_match:
        return f"-{days_match.group(1)}d"

    # Check for specific minute mentions
    minutes_match = re.search(TIME_PATTERNS['minutes'], instruction_lower)
    if minutes_match:
        return f"-{minutes_match.group(1)}m"

    # Check for specific week mentions
    weeks_match = re.search(TIME_PATTERNS['weeks'], instruction_lower)
    if weeks_match:
        return f"-{weeks_match.group(1)}w"

    # Check for specific month mentions
    months_match = re.search(TIME_PATTERNS['months'], instruction_lower)
    if months_match:
        return f"-{months_match.group(1)}mon"

    # Check for "last day/hour/week/month" (singular)
    if re.search(TIME_PATTERNS['last_day'], instruction_lower):
        return "-1d"
    if re.search(TIME_PATTERNS['last_hour'], instruction_lower):
        return "-1h"
    if re.search(TIME_PATTERNS['last_week'], instruction_lower):
        return "-1w"
    if re.search(TIME_PATTERNS['last_month'], instruction_lower):
        return "-1mon"

    return None


def substitute_time_param(query: str, new_time: str) -> str:
    """
    Substitute the time parameter in a Splunk query.

    Args:
        query: The Splunk query
        new_time: The new time parameter (e.g., '-48h')

    Returns:
        Query with updated time parameter
    """
    # Remove the leading dash for the replacement
    time_value = new_time.lstrip('-')

    # Replace earliest=-XXX pattern
    updated = re.sub(r'earliest=-\S+', f'earliest=-{time_value}', query)

    return updated


def extract_count_param(instruction: str) -> Optional[int]:
    """
    Extract count/limit parameter from instruction.

    Args:
        instruction: The user's instruction

    Returns:
        Count value or None
    """
    instruction_lower = instruction.lower()

    # Match patterns like "top 10", "first 5", "limit 20"
    patterns = [
        r'top\s+(\d+)',
        r'first\s+(\d+)',
        r'limit\s+(\d+)',
        r'(\d+)\s+results?',
        r'show\s+(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            return int(match.group(1))

    return None


def substitute_count_param(query: str, new_count: int) -> str:
    """
    Substitute count/limit parameters in a Splunk query.

    Args:
        query: The Splunk query
        new_count: The new count value

    Returns:
        Query with updated count
    """
    # Replace head X pattern
    updated = re.sub(r'\|\s*head\s+\d+', f'| head {new_count}', query)

    # Replace limit=X pattern
    updated = re.sub(r'limit=\d+', f'limit={new_count}', updated)

    return updated

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Semantic cache disabled.")


@dataclass
class CachedQuery:
    """Represents a cached approved query."""
    query_id: int
    instruction: str
    generated_query: str
    embedding: np.ndarray
    user_id: int
    corrected_query: Optional[str] = None  # If analyst provided a correction


@dataclass
class CacheMatch:
    """Result of a cache lookup."""
    found: bool
    query: Optional[str] = None
    original_instruction: str = ""
    similarity_score: float = 0.0
    query_id: Optional[int] = None
    was_corrected: bool = False
    params_modified: bool = False  # True if parameters were adjusted
    modifications: Optional[List[str]] = None  # List of modifications made


class SemanticCache:
    """
    Semantic cache for approved queries.

    Uses sentence embeddings to find queries that are semantically similar
    to the user's question. Only queries with 'good' feedback are cached.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        device: str = "cpu"
    ):
        """
        Initialize the semantic cache.

        Args:
            model_name: Sentence transformer model to use
            similarity_threshold: Minimum similarity score to return a cached result
            device: Device to run embeddings on ('cpu' or 'cuda')
        """
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.model_name = model_name
        self.model = None
        self.cache: List[CachedQuery] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Semantic cache disabled - sentence-transformers not available")
            return

        try:
            logger.info(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if the semantic cache is available and ready."""
        return self.model is not None

    def load_approved_queries(self, db_session) -> int:
        """
        Load all approved queries from the database into the cache.

        Args:
            db_session: SQLAlchemy database session

        Returns:
            Number of queries loaded into cache
        """
        if not self.is_available():
            return 0

        from src.database.models import Query, Feedback

        try:
            # Query for all queries with 'good' feedback
            approved_queries = db_session.query(
                Query.id,
                Query.instruction,
                Query.generated_query,
                Query.user_id,
                Feedback.corrected_query
            ).join(
                Feedback, Query.id == Feedback.query_id
            ).filter(
                Feedback.rating == 'good'
            ).all()

            if not approved_queries:
                logger.info("No approved queries found in database")
                self.cache = []
                self.embeddings_matrix = None
                return 0

            # Generate embeddings for all instructions
            instructions = [q.instruction for q in approved_queries]
            logger.info(f"Generating embeddings for {len(instructions)} approved queries")
            embeddings = self.model.encode(instructions, convert_to_numpy=True, show_progress_bar=False)

            # Build cache
            self.cache = []
            for i, q in enumerate(approved_queries):
                # Use corrected query if available, otherwise use generated query
                final_query = q.corrected_query if q.corrected_query else q.generated_query

                self.cache.append(CachedQuery(
                    query_id=q.id,
                    instruction=q.instruction,
                    generated_query=final_query,
                    embedding=embeddings[i],
                    user_id=q.user_id,
                    corrected_query=q.corrected_query
                ))

            # Store embeddings as matrix for efficient similarity computation
            self.embeddings_matrix = embeddings

            logger.info(f"Loaded {len(self.cache)} approved queries into semantic cache")
            return len(self.cache)

        except Exception as e:
            logger.error(f"Error loading approved queries: {e}")
            return 0

    def search(self, instruction: str) -> CacheMatch:
        """
        Search for a semantically similar approved query.

        Args:
            instruction: The user's query instruction

        Returns:
            CacheMatch with the result (found=True if match above threshold)
        """
        if not self.is_available() or len(self.cache) == 0:
            return CacheMatch(found=False)

        try:
            # Generate embedding for the query
            query_embedding = self.model.encode([instruction], convert_to_numpy=True)[0]

            # Compute cosine similarity with all cached embeddings
            similarities = self._cosine_similarity(query_embedding, self.embeddings_matrix)

            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            logger.debug(f"Best semantic match score: {best_score:.3f} (threshold: {self.similarity_threshold})")

            if best_score >= self.similarity_threshold:
                cached = self.cache[best_idx]
                logger.info(f"Cache hit! Score: {best_score:.3f}, Query ID: {cached.query_id}")

                # Apply parameter substitutions based on user's instruction
                final_query = cached.generated_query
                modifications = []

                # Extract and substitute time parameter
                user_time = extract_time_param(instruction)
                cached_time = extract_time_param(cached.instruction)

                if user_time and user_time != cached_time:
                    old_query = final_query
                    final_query = substitute_time_param(final_query, user_time)
                    if final_query != old_query:
                        modifications.append(f"Time range updated to {user_time}")
                        logger.info(f"Substituted time parameter: {cached_time} -> {user_time}")

                # Extract and substitute count parameter
                user_count = extract_count_param(instruction)
                cached_count = extract_count_param(cached.instruction)

                if user_count and user_count != cached_count:
                    old_query = final_query
                    final_query = substitute_count_param(final_query, user_count)
                    if final_query != old_query:
                        modifications.append(f"Result limit updated to {user_count}")
                        logger.info(f"Substituted count parameter: {cached_count} -> {user_count}")

                return CacheMatch(
                    found=True,
                    query=final_query,
                    original_instruction=cached.instruction,
                    similarity_score=float(best_score),
                    query_id=cached.query_id,
                    was_corrected=cached.corrected_query is not None,
                    params_modified=len(modifications) > 0,
                    modifications=modifications if modifications else None
                )

            return CacheMatch(found=False, similarity_score=float(best_score))

        except Exception as e:
            logger.error(f"Error searching semantic cache: {e}")
            return CacheMatch(found=False)

    def add_to_cache(self, query_id: int, instruction: str, generated_query: str,
                     user_id: int, corrected_query: Optional[str] = None) -> bool:
        """
        Add a newly approved query to the cache.

        Args:
            query_id: Database ID of the query
            instruction: The user's instruction
            generated_query: The generated (or corrected) query
            user_id: ID of the user who created the query
            corrected_query: Analyst's correction if provided

        Returns:
            True if successfully added
        """
        if not self.is_available():
            return False

        try:
            # Generate embedding
            embedding = self.model.encode([instruction], convert_to_numpy=True)[0]

            # Use corrected query if available
            final_query = corrected_query if corrected_query else generated_query

            # Add to cache
            cached_query = CachedQuery(
                query_id=query_id,
                instruction=instruction,
                generated_query=final_query,
                embedding=embedding,
                user_id=user_id,
                corrected_query=corrected_query
            )
            self.cache.append(cached_query)

            # Update embeddings matrix
            if self.embeddings_matrix is None:
                self.embeddings_matrix = embedding.reshape(1, -1)
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, embedding])

            logger.info(f"Added query {query_id} to semantic cache. Cache size: {len(self.cache)}")
            return True

        except Exception as e:
            logger.error(f"Error adding to cache: {e}")
            return False

    def remove_from_cache(self, query_id: int) -> bool:
        """
        Remove a query from the cache (e.g., if feedback changed to 'bad').

        Args:
            query_id: Database ID of the query to remove

        Returns:
            True if found and removed
        """
        for i, cached in enumerate(self.cache):
            if cached.query_id == query_id:
                self.cache.pop(i)
                # Rebuild embeddings matrix
                if len(self.cache) > 0:
                    self.embeddings_matrix = np.vstack([c.embedding for c in self.cache])
                else:
                    self.embeddings_matrix = None
                logger.info(f"Removed query {query_id} from semantic cache")
                return True
        return False

    def _cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query vector and all vectors in matrix.

        Args:
            query_vec: Query embedding vector
            matrix: Matrix of cached embeddings

        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)

        # Compute dot product (cosine similarity for normalized vectors)
        return np.dot(matrix_norm, query_norm)

    def get_cache_stats(self) -> dict:
        """Get statistics about the cache."""
        return {
            "available": self.is_available(),
            "model_name": self.model_name,
            "cache_size": len(self.cache),
            "similarity_threshold": self.similarity_threshold,
            "device": self.device
        }


# Global semantic cache instance
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> Optional[SemanticCache]:
    """Get the global semantic cache instance."""
    return _semantic_cache


def initialize_semantic_cache(
    similarity_threshold: float = 0.85,
    device: str = "cpu"
) -> SemanticCache:
    """
    Initialize the global semantic cache.

    Args:
        similarity_threshold: Minimum similarity for cache hits
        device: Device for embeddings ('cpu' or 'cuda')

    Returns:
        The initialized SemanticCache instance
    """
    global _semantic_cache
    _semantic_cache = SemanticCache(
        similarity_threshold=similarity_threshold,
        device=device
    )
    return _semantic_cache
