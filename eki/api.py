import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
from fastapi.responses import FileResponse, Response
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from .config import Settings
from .models import ChatRequest, ChatResponse, SearchRequest, WebRequest
from .ingest import fetch_web
from .service import Service


def create_app(settings=None):
    settings = settings or Settings()
    registry = CollectorRegistry()
    chats = Counter('eki_chat_requests', 'Chat requests', ['status', 'cached'], registry=registry)
    latency = Histogram('eki_chat_seconds', 'End-to-end chat latency', registry=registry)
    failures = Counter('eki_errors', 'Request errors', ['operation'], registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        logging.basicConfig(level=logging.INFO)
        app.state.service = Service(settings)
        yield
        app.state.service.close()

    app = FastAPI(title='Enterprise Knowledge Intelligence', version='0.1.0', lifespan=lifespan)

    def auth(x_api_key: str = Header(default='')):
        if not secrets.compare_digest(x_api_key, settings.api_key):
            raise HTTPException(401, 'Invalid API key')

    @app.get('/', include_in_schema=False)
    def home():
        return FileResponse(Path(__file__).parent / 'static/index.html')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'version': '0.1.0'}

    @app.get('/status', dependencies=[Depends(auth)])
    def status():
        service = app.state.service
        return {'chunks': len(service.index.chunks), 'embedding': settings.embedding,
                'reranker': settings.reranker, 'generator': settings.generator,
                'cache': 'redis' if settings.redis_url else 'local-ttl',
                'documents': sorted({c.source for c in service.index.chunks})}

    @app.post('/documents/upload', dependencies=[Depends(auth)])
    def upload(file: UploadFile = File(...), strategy: str = Form('recursive')):
        data = file.file.read(settings.max_upload_bytes + 1)
        try:
            return app.state.service.ingest(file.filename or 'document', data, strategy)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            failures.labels('upload').inc()
            raise HTTPException(422, 'Document processing failed; check format and configured services') from exc
        finally:
            file.file.close()

    @app.post('/documents/web', dependencies=[Depends(auth)])
    def web(request: WebRequest):
        try:
            data = fetch_web(request.url, settings.web_hosts, settings.max_upload_bytes)
            return app.state.service.ingest('web.html', data, source_override=request.url)
        except Exception as exc:
            failures.labels('web').inc()
            raise HTTPException(422, 'Web import rejected or unavailable; check host allowlist and HTML limits') from exc

    @app.post('/search', dependencies=[Depends(auth)])
    def search(request: SearchRequest):
        with app.state.service.lock:
            return {'results': app.state.service.index.search(request.query, request.limit)}

    @app.post('/chat', response_model=ChatResponse, dependencies=[Depends(auth)])
    def chat(request: ChatRequest):
        try:
            with latency.time():
                response = app.state.service.chat(request.question, request.session_id)
            chats.labels(response.status, str(response.cached).lower()).inc()
            return response
        except Exception as exc:
            failures.labels('chat').inc()
            logging.getLogger('eki').error('Chat failed: %s', type(exc).__name__)
            raise HTTPException(503, 'A required model or service is unavailable; no answer was fabricated') from exc

    @app.get('/evidence/{chunk_id}', dependencies=[Depends(auth)])
    def evidence(chunk_id: str):
        with app.state.service.lock:
            chunk = app.state.service.index.by_id.get(chunk_id)
            if chunk is None:
                raise HTTPException(404, 'Evidence not found')
            return chunk

    @app.get('/metrics', dependencies=[Depends(auth)])
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
