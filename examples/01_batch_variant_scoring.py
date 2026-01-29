"""Example: Batch variant scoring with caching and GPU acceleration.

This example demonstrates how to efficiently score thousands of variants
using AlphaGenome-Plus's batch processing and caching capabilities.
"""

import asyncio
from alphagenome.data import genome
from alphagenome_plus.batch import BatchVariantProcessor, BatchConfig
from alphagenome_plus.cache import PredictionCache
from alphagenome_plus.acceleration import GPUAccelerator


async def main():
    # Configuration
    API_KEY = 'your_api_key_here'
    
    # Initialize components
    cache = PredictionCache(cache_dir="./variant_cache")
    gpu = GPUAccelerator(device='cuda')
    
    config = BatchConfig(
        max_concurrent=5,
        retry_attempts=3,
        rate_limit_per_minute=100,
        use_cache=True
    )
    
    processor = BatchVariantProcessor(
        api_key=API_KEY,
        config=config,
        cache=cache,
        accelerator=gpu
    )
    
    # Define genomic interval
    interval = genome.Interval(
        chromosome='chr22',
        start=35677410,
        end=36725986
    )
    
    # Load variants from VCF or create programmatically
    variants = [
        genome.Variant(
            chromosome='chr22',
            position=36201698 + i * 1000,
            reference_bases='A',
            alternate_bases='T'
        )
        for i in range(100)  # 100 variants
    ]
    
    print(f"Processing {len(variants)} variants...")
    
    # Batch process with progress tracking
    results = await processor.process_variants(
        interval=interval,
        variants=variants,
        ontology_terms=['UBERON:0001157'],
        show_progress=True
    )
    
    # Analyze results
    print(f"\nSuccessfully processed: {len(results)} variants")
    print(f"Cache hit rate: {cache.get_hit_rate():.2%}")
    
    # Extract pathogenicity scores
    scores = []
    for variant, result in results.items():
        if result is not None:
            score = gpu.compute_variant_effect_score(result)
            scores.append((variant.position, score))
            
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most impactful variants:")
    for position, score in scores[:10]:
        print(f"  Position {position}: score = {score:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
