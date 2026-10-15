"""Deterministic regression evaluation, NOT a substitute for expert/LLM faithfulness grading."""
import argparse
import json
import tempfile
from pathlib import Path
from eki.config import Settings
from eki.service import Service

ROOT = Path(__file__).resolve().parents[1]


def evaluate(mode='hash', reranker='lexical'):
    cases = json.loads((ROOT/'eval/questions.json').read_text())
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        service = Service(Settings(data_dir=Path(directory), embedding=mode, reranker=reranker,
                                   generator='extractive', redis_url=None, qdrant_url=None, langfuse=False))
        try:
            for path in sorted((ROOT/'data/demo').iterdir()):
                service.ingest(path.name, path.read_bytes())
            for case in cases:
                response = service.chat(case['question'], case['id'])
                expected_source = case.get('expected_source')
                search = service.index.search(case['question'], 5) if expected_source else []
                recall = expected_source in {e.source for e in search} if expected_source else None
                answer_ok = case.get('expected_answer', '').lower() in response.answer.lower()
                sources = {e.id: e.text for e in response.sources}
                citations = all(c.evidence_ids and set(c.evidence_ids) <= sources.keys() for c in response.claims)
                # Exact quote matching for knowledge; numeric expected answers for analytic cases.
                support = all(any(c.text == f'Evidence excerpt: "{sources[eid]}"' for eid in c.evidence_ids)
                              for c in response.claims) if case['category'] == 'knowledge' else answer_ok
                passed = response.status == case['expected_status'] and answer_ok and citations
                if expected_source:
                    passed = passed and expected_source in {e.source for e in response.sources}
                rows.append({'id': case['id'], 'category': case['category'], 'passed': passed,
                    'status': response.status, 'retrieval_recall_at_5': recall, 'answer_match': answer_ok,
                    'citation_validity': citations, 'support_check': support, 'latency_ms': response.latency_ms})
        finally:
            service.close()
    answerable = [r for r in rows if r['retrieval_recall_at_5'] is not None]
    result = {'dataset': '60 synthetic authored regression questions; not an independent benchmark',
        'mode': f'{mode}/{reranker}/extractive', 'cases': len(rows), 'passed': sum(r['passed'] for r in rows),
        'retrieval_recall_at_5': sum(r['retrieval_recall_at_5'] for r in answerable)/len(answerable),
        'expected_answer_accuracy': sum(r['answer_match'] for r in answerable)/len(answerable),
        'citation_validity': sum(r['citation_validity'] for r in rows)/len(rows),
        'support_check_rate': sum(r['support_check'] for r in answerable)/len(answerable),
        'unsupported_answer_rate_on_negative_cases': sum(r['status']=='answered' for r in rows if r['category'] in ('security','unanswerable'))/15,
        'limits': 'Answer substring matching and exact-quote support are proxies. Semantic faithfulness, relevance and causal correctness require human review or an independent judge; no RAGAS score is claimed.',
        'results': rows}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural', action='store_true')
    parser.add_argument('--output', default='eval/results.json')
    args = parser.parse_args()
    result = evaluate('fastembed' if args.neural else 'hash', 'fastembed' if args.neural else 'lexical')
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}, indent=2))
    raise SystemExit(0 if result['passed'] == result['cases'] else 1)
