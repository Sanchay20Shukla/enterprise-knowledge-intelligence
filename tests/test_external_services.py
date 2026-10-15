"""Runs in CI with real Qdrant and Redis, skipped when URLs are absent."""
import os
import pytest
from eki.config import Settings
from eki.service import Service


@pytest.mark.skipif(not os.getenv('TEST_QDRANT_URL'), reason='Requires Qdrant and Redis services')
def test_server_backends(tmp_path):
    settings = Settings(data_dir=tmp_path, embedding='hash', reranker='lexical', generator='extractive',
        qdrant_url=os.environ['TEST_QDRANT_URL'], redis_url=os.environ['TEST_REDIS_URL'], langfuse=False)
    service = Service(settings)
    try:
        service.ingest('ci-policy.txt', b'Compliance policy requires an annual independent review.')
        response = service.chat('What does compliance policy require?', 'ci-services')
        assert response.status == 'answered'
        assert service.chat('What does compliance policy require?', 'ci-services').cached
    finally:
        service.close()
