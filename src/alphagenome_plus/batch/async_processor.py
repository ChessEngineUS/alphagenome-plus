"""Asynchronous batch processing for AlphaGenome predictions.

Provides efficient parallel processing of large variant sets with
rate limiting, error handling, and progress tracking.
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
import time
from collections import deque
import numpy as np
from tqdm.asyncio import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_concurrent: int = 10  # Max concurrent requests
    rate_limit: float = 5.0  # Requests per second
    retry_attempts: int = 3
    retry_delay: float = 1.0  # Initial retry delay (exponential backoff)
    timeout: float = 30.0  # Request timeout in seconds
    cache_results: bool = True


@dataclass
class BatchResult:
    """Result from batch processing."""
    successful: List[Dict] = field(default_factory=list)
    failed: List[Dict] = field(default_factory=list)
    total_time: float = 0.0
    success_rate: float = 0.0


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, rate: float):
        """Initialize rate limiter.
        
        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire token for request (blocks if rate exceeded)."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Refill tokens based on elapsed time
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            # Wait if no tokens available
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 1.0
                self.last_update = time.time()
            
            self.tokens -= 1.0


class AsyncBatchProcessor:
    """Asynchronous batch processor for AlphaGenome predictions.
    
    Args:
        model: AlphaGenome DNA client
        config: Batch processing configuration
        progress_callback: Optional callback for progress updates
    """
    
    def __init__(self, model: Any, 
                 config: Optional[BatchConfig] = None,
                 progress_callback: Optional[Callable] = None):
        self.model = model
        self.config = config or BatchConfig()
        self.progress_callback = progress_callback
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        self.cache = {} if self.config.cache_results else None
    
    async def _process_single(self, task: Dict, 
                            semaphore: asyncio.Semaphore) -> Dict:
        """Process single prediction task with error handling.
        
        Args:
            task: Task dictionary with 'interval', 'variant', etc.
            semaphore: Semaphore for concurrency control
            
        Returns:
            Result dictionary with success/error information
        """
        async with semaphore:
            # Check cache
            cache_key = str(task) if self.cache is not None else None
            if cache_key and cache_key in self.cache:
                return self.cache[cache_key]
            
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Retry logic
            last_error = None
            for attempt in range(self.config.retry_attempts):
                try:
                    # Execute prediction
                    result = await self._execute_prediction(task)
                    
                    # Cache successful result
                    if self.cache is not None:
                        self.cache[cache_key] = result
                    
                    return {'status': 'success', 'task': task, 'result': result}
                
                except Exception as e:
                    last_error = e
                    if attempt < self.config.retry_attempts - 1:
                        # Exponential backoff
                        delay = self.config.retry_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        logger.warning(f"Retry {attempt+1}/{self.config.retry_attempts}: {e}")
            
            # All retries failed
            return {
                'status': 'failed',
                'task': task,
                'error': str(last_error)
            }
    
    async def _execute_prediction(self, task: Dict) -> Any:
        """Execute single prediction (runs in thread pool to avoid blocking).
        
        Args:
            task: Task specification
            
        Returns:
            Prediction result
        """
        loop = asyncio.get_event_loop()
        
        # Run synchronous model.predict() in thread pool
        if 'variant' in task:
            result = await loop.run_in_executor(
                None,
                lambda: self.model.predict_variant(
                    interval=task['interval'],
                    variant=task['variant'],
                    ontology_terms=task.get('ontology_terms', []),
                    requested_outputs=task.get('requested_outputs', [])
                )
            )
        else:
            result = await loop.run_in_executor(
                None,
                lambda: self.model.predict(
                    interval=task['interval'],
                    ontology_terms=task.get('ontology_terms', []),
                    requested_outputs=task.get('requested_outputs', [])
                )
            )
        
        return result
    
    async def process_batch(self, tasks: List[Dict]) -> BatchResult:
        """Process batch of prediction tasks asynchronously.
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            BatchResult with successful and failed predictions
        """
        start_time = time.time()
        
        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Create coroutines for all tasks
        coroutines = [
            self._process_single(task, semaphore) 
            for task in tasks
        ]
        
        # Execute with progress bar
        results = []
        for coro in tqdm.as_completed(coroutines, total=len(tasks),
                                      desc="Processing batch"):
            result = await coro
            results.append(result)
            
            # Progress callback
            if self.progress_callback:
                self.progress_callback(len(results), len(tasks))
        
        # Separate successful and failed
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        # Compute statistics
        total_time = time.time() - start_time
        success_rate = len(successful) / len(tasks) if tasks else 0.0
        
        logger.info(f"Batch complete: {len(successful)}/{len(tasks)} successful ")
        logger.info(f"Time: {total_time:.2f}s, Rate: {len(tasks)/total_time:.2f} req/s")
        
        return BatchResult(
            successful=successful,
            failed=failed,
            total_time=total_time,
            success_rate=success_rate
        )
    
    def process_batch_sync(self, tasks: List[Dict]) -> BatchResult:
        """Synchronous wrapper for process_batch.
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            BatchResult
        """
        return asyncio.run(self.process_batch(tasks))


class BatchVariantScorer:
    """High-level interface for batch variant scoring.
    
    Args:
        model: AlphaGenome DNA client
        config: Batch processing configuration
    """
    
    def __init__(self, model: Any, config: Optional[BatchConfig] = None):
        self.processor = AsyncBatchProcessor(model, config)
    
    def score_variants(self, variants: List[Dict],
                      interval_size: int = 100000) -> List[Dict]:
        """Score batch of variants for pathogenicity.
        
        Args:
            variants: List of variant dicts with 'chromosome', 'position', 
                     'reference_bases', 'alternate_bases'
            interval_size: Size of genomic interval around variant
            
        Returns:
            List of scored variants
        """
        # Build tasks
        tasks = []
        for v in variants:
            from alphagenome.data import genome
            
            # Define interval around variant
            start = max(0, v['position'] - interval_size // 2)
            end = v['position'] + interval_size // 2
            
            interval = genome.Interval(
                chromosome=v['chromosome'],
                start=start,
                end=end
            )
            
            variant = genome.Variant(
                chromosome=v['chromosome'],
                position=v['position'],
                reference_bases=v['reference_bases'],
                alternate_bases=v['alternate_bases']
            )
            
            tasks.append({
                'interval': interval,
                'variant': variant,
                'ontology_terms': v.get('ontology_terms', []),
                'requested_outputs': v.get('requested_outputs', [])
            })
        
        # Process batch
        results = self.processor.process_batch_sync(tasks)
        
        # Extract scores
        scored_variants = []
        for i, res in enumerate(results.successful):
            variant_dict = variants[i].copy()
            
            # Compute pathogenicity score from predictions
            # (simplified - would use actual scoring logic)
            variant_dict['pathogenicity_score'] = self._compute_score(res['result'])
            scored_variants.append(variant_dict)
        
        return scored_variants
    
    def _compute_score(self, prediction_output: Any) -> float:
        """Compute pathogenicity score from prediction.
        
        Args:
            prediction_output: AlphaGenome prediction result
            
        Returns:
            Pathogenicity score 0-1
        """
        # Simplified scoring - would integrate multiple signals
        # Compare reference vs alternate predictions
        
        score = 0.5  # Neutral baseline
        
        try:
            # RNA expression difference
            if hasattr(prediction_output, 'reference') and hasattr(prediction_output, 'alternate'):
                ref_expr = np.mean(prediction_output.reference.rna_seq.data)
                alt_expr = np.mean(prediction_output.alternate.rna_seq.data)
                
                expr_diff = abs(alt_expr - ref_expr) / (ref_expr + 1e-8)
                score += min(expr_diff * 0.3, 0.3)
            
            # Chromatin accessibility difference
            if hasattr(prediction_output, 'chromatin_features'):
                # Increased score for changes in regulatory regions
                score += 0.2
        
        except Exception as e:
            logger.warning(f"Error computing score: {e}")
        
        return min(score, 1.0)
