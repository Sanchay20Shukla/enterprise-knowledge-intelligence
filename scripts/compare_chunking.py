import json
from pathlib import Path
from eki.ingest import chunk_documents, ENCODER
from eki.models import Document
from eki.retrieval import Embedder

embedder = Embedder('fastembed')
document = Document(source='chunking-experiment', document_type='synthetic', content=(
    'Employees can work remotely three days per week. Managers approve the weekly remote schedule. '
    'Annual leave requests need ten days of notice. Human resources records the leave balance.\n\n'
    'Database backups run every night. Restoration tests are performed monthly. '
    'Privileged access is reviewed quarterly by system owners. Security incidents require reporting within two hours.\n\n'
    'Quarterly revenue fell after delayed renewals. Management expects renewal activity to recover next quarter. '
    'Procurement requires supplier quotations for large purchases. The finance team reviews the purchase order.'))
results = {}
for strategy in ('recursive', 'token', 'semantic'):
    chunks = chunk_documents([document], 'experiment', strategy, max_tokens=70, embedder=embedder)
    results[strategy] = {'count': len(chunks), 'token_counts': [len(ENCODER.encode(c.content)) for c in chunks],
                         'chunks': [c.content for c in chunks]}
Path('eval/chunking-comparison.json').write_text(json.dumps(results, indent=2))
print({strategy: result['count'] for strategy, result in results.items()})
