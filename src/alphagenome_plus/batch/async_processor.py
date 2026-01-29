"""Async batch processor with advanced features."""

import asyncio
from typing import List, Callable, Any, Optional, TypeVar
from dataclasses import dataclass
import logging

T = TypeVar('T')
R = TypeVar('R')

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of async processing."""
    item: Any
    result: Optional[Any] = None
    error: Optional[Exception] = None
    retry_count: int = 0


class AsyncBatchProcessor:
    """Generic async batch processor with retry logic.
    
    Example:
        >>> processor = AsyncBatchProcessor(max_concurrent=50, max_retries=3)
        >>> results = await processor.process_batch(
        ...     items=variants,
        ...     process_func=predict_single_variant,
        ...     show_progress=True
        ... )
    """
    
    def __init__(
        self,
        max_concurrent: int = 50,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True,
    ):
        """Initialize processor.
        
        Args:
            max_concurrent: Maximum concurrent tasks
            max_retries: Maximum retry attempts per item
            retry_delay: Base delay between retries (seconds)
            exponential_backoff: Use exponential backoff for retries
        """
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(
        self,
        item: T,
        process_func: Callable[[T], R],
    ) -> ProcessingResult:
        """Process single item with retry logic.
        
        Args:
            item: Item to process
            process_func: Processing function
            
        Returns:
            ProcessingResult with outcome
        """
        async with self.semaphore:
            for retry in range(self.max_retries + 1):
                try:
                    result = await process_func(item)
                    return ProcessingResult(item=item, result=result, retry_count=retry)
                except Exception as e:
                    if retry == self.max_retries:
                        logger.error(f"Failed to process {item} after {retry} retries: {e}")
                        return ProcessingResult(item=item, error=e, retry_count=retry)
                    
                    # Calculate delay with optional exponential backoff
                    delay = self.retry_delay
                    if self.exponential_backoff:
                        delay *= 2 ** retry
                    
                    logger.warning(f"Retry {retry + 1}/{self.max_retries} for {item} after {delay}s")
                    await asyncio.sleep(delay)
    
    async def process_batch(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        show_progress: bool = False,
    ) -> List[ProcessingResult]:
        """Process batch of items concurrently.
        
        Args:
            items: List of items to process
            process_func: Async processing function
            show_progress: Show progress bar
            
        Returns:
            List of ProcessingResults
        """
        tasks = [self.process_item(item, process_func) for item in items]
        
        if show_progress:
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc="Processing")
        else:
            results = await asyncio.gather(*tasks)
        
        return results