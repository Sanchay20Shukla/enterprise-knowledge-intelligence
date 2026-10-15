import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import PurePath
from .ingest import parse_file, chunk_documents
from .models import ChatResponse
from .retrieval import Index
from .sql_agent import SQLAgent
from .state import StateStore
from .workflow import Workflow

log = logging.getLogger("eki")


class Service:
    def __init__(self, settings):
        self.settings = settings
        self.lock = threading.RLock()
        self.index = Index(settings)
        self.sql = SQLAgent(settings.data_dir / 'analytics.sqlite')
        self.state = StateStore(settings.redis_url)
        self.workflow = Workflow(self.index, self.sql, settings)

    def ingest(self, name, data, strategy="recursive", source_override=None):
        if len(data) > self.settings.max_upload_bytes:
            raise ValueError("Upload exceeds size limit")
        docs = parse_file(name, data, self.settings.max_text_chars)
        if source_override:
            for doc in docs:
                doc.source = source_override
        # Include source to preserve separately named identical documents. Replacing an existing name
        # is deliberately disallowed: avoids stale chunks and ambiguous version provenance.
        source = docs[0].source
        doc_id = hashlib.sha256(source.encode() + b'\0' + data).hexdigest()
        chunks = chunk_documents(docs, doc_id, strategy, embedder=self.index.embedder)
        if len(chunks) > self.settings.max_chunks:
            raise ValueError("Document exceeds chunk limit")
        with self.lock:
            existing = [c for c in self.index.chunks if c.source == source]
            if existing and any(c.document_id != doc_id for c in existing):
                raise ValueError("Source name already exists with different content; use a versioned filename")
            if len(self.index.chunks) + len([c for c in chunks if c.id not in self.index.by_id]) > self.settings.max_corpus_chunks:
                raise ValueError("Corpus limit reached")
            # Validate analytics schema before committing chunks.
            if PurePath(name).suffix.lower() == '.csv':
                self.sql.import_csv(source, data)
            count = self.index.add(chunks)
        return {"document_id": doc_id, "chunks_added": count, "source": source, "strategy": strategy}

    def chat(self, question, session_id):
        start = time.perf_counter()
        trace = str(uuid.uuid4())
        with self.lock:
            memory_key = 'session:' + session_id
            memory = self.state.get(memory_key) or []
            # Bounded context only resolves simple explicit follow-ups, never appends assistant text.
            resolved = question
            if question.lower().startswith(('why did it', 'what about that', 'explain that')) and memory:
                resolved = memory[-1] + '\nFollow-up: ' + question
            fingerprint = {'q': resolved, 'revision': self.index.revision, 'generator': self.settings.generator,
                           'model': self.settings.llm_model, 'reranker': self.settings.reranker, 'version': '1'}
            key = 'answer:' + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
            cached = self.state.get(key)
            if cached:
                response = ChatResponse.model_validate(cached)
                response.cached, response.trace_id = True, trace
                response.usage = {}  # No model call on a hit; do not double-count cached token usage.
            else:
                observation = nullcontext()
                if self.settings.langfuse:
                    from langfuse import get_client
                    observation = get_client().start_as_current_observation(name='enterprise-rag',
                        metadata={'request_id': trace, 'mode': self.settings.generator})
                with observation as span:
                    result = self.workflow.graph.invoke({'question': resolved})
                    cited = {eid for c in result['claims'] for eid in c.evidence_ids}
                    sources = [e for e in result['evidence'] if e.id in cited]
                    response = ChatResponse(answer=result['answer'], claims=result['claims'],
                        confidence=1.0 if sources and self.settings.generator == 'extractive' else 0.5 if sources else 0,
                        sources=sources, tools_used=result['tools'], status=result['status'],
                        mode=f'{self.settings.embedding}/{self.settings.reranker}/{self.settings.generator}',
                        trace_id=trace, usage=result['usage'])
                    if span:
                        span.update(metadata={'request_id': trace, 'status': response.status,
                            'evidence_ids': [e.id for e in sources], 'tools': response.tools_used,
                            'usage': response.usage, 'elapsed_ms': (time.perf_counter()-start)*1000})
                self.state.put(key, response.model_dump(), self.settings.cache_ttl)
            if response.status != 'blocked':
                self.state.put(memory_key, (memory + [question])[-4:], self.settings.session_ttl)
            response.latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log.info(json.dumps({'event': 'chat', 'request_id': trace, 'status': response.status,
                'cache_hit': response.cached, 'latency_ms': response.latency_ms, 'tools': response.tools_used,
                'evidence_ids': [e.id for e in response.sources], 'usage': response.usage}))
            return response

    def close(self):
        self.index.close()
        self.state.close()
