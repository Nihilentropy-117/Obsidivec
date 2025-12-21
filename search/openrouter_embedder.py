"""
OpenRouter API client for generating embeddings.
Provides batch processing, error handling, and retry logic.
"""

import logging
import time
from typing import List
import requests


logger = logging.getLogger("openrouter-embedder")


class OpenRouterAPIError(Exception):
    """Base exception for OpenRouter API errors."""
    pass


class OpenRouterRateLimitError(OpenRouterAPIError):
    """Raised when API rate limit is exceeded."""
    pass


class OpenRouterAuthError(OpenRouterAPIError):
    """Raised when API authentication fails."""
    pass


class OpenRouterEmbedder:
    """
    OpenRouter API client for generating embeddings.

    Features:
    - Batch processing to optimize API calls
    - Exponential backoff retry logic
    - Rate limit handling
    - Comprehensive error handling
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/text-embedding-3-small",
        batch_size: int = 25,
        timeout: int = 30
    ):
        """
        Initialize the OpenRouter embedder.

        Args:
            api_key: OpenRouter API key
            model: Embedding model to use
            batch_size: Number of texts to process per API request
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.api_url = "https://openrouter.ai/api/v1/embeddings"

        if not self.api_key:
            raise OpenRouterAuthError("API key is required")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            OpenRouterAPIError: If API request fails after retries
            OpenRouterAuthError: If authentication fails
            OpenRouterRateLimitError: If rate limit is exceeded
        """
        if not texts:
            return []

        # Process in batches
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            logger.info(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} texts)"
            )

            batch_embeddings = self._batch_encode(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _batch_encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a single batch of texts.

        Args:
            texts: List of text strings (batch)

        Returns:
            List of embedding vectors
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response_data = self._make_api_request(texts)

                # Extract embeddings from response
                embeddings = [item['embedding'] for item in response_data['data']]

                # Validate response
                if len(embeddings) != len(texts):
                    raise OpenRouterAPIError(
                        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                    )

                return embeddings

            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Network error after {max_retries} attempts: {e}")
                    raise OpenRouterAPIError(f"Network error: {e}")

                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    f"Network error on attempt {attempt + 1}/{max_retries}, "
                    f"retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

            except OpenRouterRateLimitError:
                if attempt == max_retries - 1:
                    raise

                wait_time = 2 ** attempt
                logger.warning(f"Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)

    def _make_api_request(self, texts: List[str]) -> dict:
        """
        Make an API request to OpenRouter.

        Args:
            texts: List of text strings

        Returns:
            Response JSON data

        Raises:
            OpenRouterAuthError: If authentication fails
            OpenRouterRateLimitError: If rate limit is exceeded
            OpenRouterAPIError: If request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "input": texts
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            # Handle authentication errors
            if response.status_code == 401:
                logger.error("Authentication failed - invalid API key")
                raise OpenRouterAuthError(
                    "Invalid API key. Check OPENROUTER_API_KEY. "
                    "Get your key from https://openrouter.ai"
                )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(
                    f"Rate limited by OpenRouter API. "
                    f"Waiting {retry_after}s..."
                )
                time.sleep(retry_after)
                raise OpenRouterRateLimitError("Rate limit exceeded")

            # Handle server errors
            if 500 <= response.status_code < 600:
                logger.error(
                    f"OpenRouter server error: {response.status_code} "
                    f"{response.text}"
                )
                raise OpenRouterAPIError(
                    f"Server error: {response.status_code}"
                )

            # Raise for other HTTP errors
            response.raise_for_status()

            # Parse and validate response
            try:
                data = response.json()

                if 'data' not in data:
                    raise OpenRouterAPIError(
                        "Invalid API response: missing 'data' field"
                    )

                return data

            except ValueError as e:
                logger.error(f"Failed to parse API response: {e}")
                raise OpenRouterAPIError(f"Malformed JSON response: {e}")

        except requests.RequestException as e:
            if isinstance(e, (requests.ConnectionError, requests.Timeout)):
                raise  # Let caller handle retries
            logger.error(f"API request failed: {e}")
            raise OpenRouterAPIError(f"Request failed: {e}")
