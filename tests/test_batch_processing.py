"""Unit tests for batch processing."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from alphagenome_plus.batch.async_processor import (
    BatchConfig,
    BatchResult,
    RateLimiter,
    AsyncBatchProcessor,
    BatchVariantScorer
)


class TestBatchConfig:
    """Test batch configuration."""
    
    def test_default_config(self):
        config = BatchConfig()
        assert config.max_concurrent == 10
        assert config.rate_limit == 5.0
        assert config.retry_attempts == 3
        assert config.cache_results is True
    
    def test_custom_config(self):
        config = BatchConfig(
            max_concurrent=20,
            rate_limit=10.0,
            retry_attempts=5,
            cache_results=False
        )
        assert config.max_concurrent == 20
        assert config.rate_limit == 10.0


class TestRateLimiter:
    """Test rate limiting."""
    
    @pytest.mark.asyncio
    async def test_single_acquisition(self):
        limiter = RateLimiter(rate=10.0)  # 10 req/s
        
        # Should not block on first acquisition
        await limiter.acquire()
        assert limiter.tokens < 10.0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        limiter = RateLimiter(rate=5.0)  # 5 req/s
        
        # Acquire tokens rapidly
        start = asyncio.get_event_loop().time()
        for _ in range(6):  # More than rate
            await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start
        
        # Should take at least 1 second for 6 requests at 5 req/s
        assert elapsed >= 0.2  # Some delay expected


class TestAsyncBatchProcessor:
    """Test async batch processing."""
    
    @pytest.fixture
    def mock_model(self):
        return Mock()
    
    @pytest.fixture
    def processor(self, mock_model):
        config = BatchConfig(max_concurrent=5, rate_limit=100.0)
        return AsyncBatchProcessor(mock_model, config)
    
    @pytest.mark.asyncio
    async def test_process_single_success(self, processor, mock_model):
        # Mock successful prediction
        processor._execute_prediction = AsyncMock(
            return_value={'prediction': 'test'}
        )
        
        task = {'interval': Mock(), 'variant': Mock()}
        semaphore = asyncio.Semaphore(5)
        
        result = await processor._process_single(task, semaphore)
        
        assert result['status'] == 'success'
        assert 'result' in result
    
    @pytest.mark.asyncio
    async def test_process_single_retry_then_success(self, processor):
        # Fail twice, then succeed
        call_count = 0
        
        async def mock_execute(task):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return {'prediction': 'success'}
        
        processor._execute_prediction = mock_execute
        processor.config.retry_delay = 0.01  # Fast retries for testing
        
        task = {'interval': Mock()}
        semaphore = asyncio.Semaphore(5)
        
        result = await processor._process_single(task, semaphore)
        
        assert result['status'] == 'success'
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_process_single_all_retries_fail(self, processor):
        # Always fail
        processor._execute_prediction = AsyncMock(
            side_effect=Exception("Permanent error")
        )
        processor.config.retry_attempts = 2
        processor.config.retry_delay = 0.01
        
        task = {'interval': Mock()}
        semaphore = asyncio.Semaphore(5)
        
        result = await processor._process_single(task, semaphore)
        
        assert result['status'] == 'failed'
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_process_batch(self, processor):
        # Mock successful execution
        processor._execute_prediction = AsyncMock(
            return_value={'prediction': 'test'}
        )
        
        tasks = [
            {'interval': Mock(), 'id': i}
            for i in range(5)
        ]
        
        result = await processor.process_batch(tasks)
        
        assert isinstance(result, BatchResult)
        assert len(result.successful) == 5
        assert len(result.failed) == 0
        assert result.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_process_batch_partial_failure(self, processor):
        # Fail on certain tasks
        async def mock_execute(task):
            if task['id'] % 2 == 0:
                raise Exception("Error")
            return {'prediction': 'success'}
        
        processor._execute_prediction = mock_execute
        processor.config.retry_attempts = 1
        processor.config.retry_delay = 0.01
        
        tasks = [
            {'interval': Mock(), 'id': i}
            for i in range(6)
        ]
        
        result = await processor.process_batch(tasks)
        
        assert len(result.successful) == 3  # IDs 1, 3, 5
        assert len(result.failed) == 3  # IDs 0, 2, 4
        assert result.success_rate == 0.5
    
    def test_process_batch_sync(self, processor):
        processor._execute_prediction = AsyncMock(
            return_value={'prediction': 'test'}
        )
        
        tasks = [{'interval': Mock(), 'id': i} for i in range(3)]
        
        result = processor.process_batch_sync(tasks)
        
        assert isinstance(result, BatchResult)
        assert len(result.successful) == 3


class TestBatchVariantScorer:
    """Test high-level variant scoring."""
    
    @pytest.fixture
    def mock_model(self):
        return Mock()
    
    @pytest.fixture
    def scorer(self, mock_model):
        config = BatchConfig(max_concurrent=5, rate_limit=100.0)
        return BatchVariantScorer(mock_model, config)
    
    def test_compute_score(self, scorer):
        # Mock prediction output
        mock_output = Mock()
        mock_output.reference = Mock()
        mock_output.alternate = Mock()
        mock_output.reference.rna_seq = Mock(data=np.ones(100))
        mock_output.alternate.rna_seq = Mock(data=np.ones(100) * 1.5)
        
        score = scorer._compute_score(mock_output)
        
        assert 0 <= score <= 1.0
    
    @patch('alphagenome_plus.batch.async_processor.genome')
    def test_score_variants(self, mock_genome, scorer, mock_model):
        # Mock genome module
        mock_genome.Interval = Mock()
        mock_genome.Variant = Mock()
        
        # Mock processor
        scorer.processor.process_batch_sync = Mock(
            return_value=BatchResult(
                successful=[
                    {'result': Mock(reference=Mock(rna_seq=Mock(data=np.ones(100))),
                                  alternate=Mock(rna_seq=Mock(data=np.ones(100) * 1.5)))}
                    for _ in range(2)
                ],
                failed=[]
            )
        )
        
        variants = [
            {
                'chromosome': 'chr1',
                'position': 12345,
                'reference_bases': 'A',
                'alternate_bases': 'G'
            },
            {
                'chromosome': 'chr1',
                'position': 67890,
                'reference_bases': 'C',
                'alternate_bases': 'T'
            }
        ]
        
        scored = scorer.score_variants(variants)
        
        assert len(scored) == 2
        assert all('pathogenicity_score' in v for v in scored)
