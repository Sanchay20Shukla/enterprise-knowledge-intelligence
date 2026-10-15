import json
import statistics
import tempfile
from pathlib import Path
from eki.config import Settings
from eki.service import Service

with tempfile.TemporaryDirectory() as directory:
    service = Service(Settings(data_dir=Path(directory), embedding='hash', reranker='lexical', generator='extractive',
                               redis_url=None, qdrant_url=None, langfuse=False))
    try:
        service.ingest('policy.txt', b'Remote work permits three days per week with manager approval.')
        misses, hits = [], []
        for i in range(30):
            question = f'Explain the remote work policy. Request {i}.'
            misses.append(service.chat(question, f'bench-{i}').latency_ms)
            response = service.chat(question, f'bench-{i}')
            assert response.cached
            hits.append(response.latency_ms)
        result = {'backend': 'bounded local TTL cache', 'mode': 'hash/lexical/extractive', 'pairs': 30,
            'median_miss_ms': statistics.median(misses), 'median_hit_ms': statistics.median(hits),
            'llm_calls': 0, 'measured_api_cost': 0,
            'note': 'No paid model calls. This measures local caching, not Redis network latency or paid-model savings.'}
        Path('eval/cache-benchmark.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        service.close()
