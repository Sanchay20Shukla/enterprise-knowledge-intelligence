"""Public, read-only synthetic-corpus demo of the existing EKI service."""
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Enterprise Knowledge Assistant',page_icon='📚',layout='wide')

@st.cache_resource(show_spinner='Indexing the synthetic enterprise documents…')
def get_service():
    from eki.config import Settings
    from eki.service import Service
    settings=Settings(_env_file=None,data_dir=Path(mkdtemp(prefix='eki-public-')),
        embedding='hash',reranker='lexical',generator='extractive',
        qdrant_url=None,qdrant_api_key=None,redis_url=None,langfuse=False)
    service=Service(settings)
    try:
        for path in sorted((ROOT/'data/demo').iterdir()):
            if path.is_file():
                service.ingest(path.name,path.read_bytes())
    except Exception:
        service.close()
        raise
    return service

def display_response(result):
    st.markdown(result['answer'])
    st.caption(f"Status: {result['status']} · {result['latency_ms']:,.0f} ms · "
               f"Cache: {'hit' if result['cached'] else 'miss'} · Mode: {result['mode']}")
    if result['tools_used']:
        st.caption('Workflow: '+' → '.join(result['tools_used']))
    for source in result['sources']:
        location=f" · page {source['page']}" if source.get('page') else f" · row {source['row']}" if source.get('row') else ''
        with st.expander(source['source']+location+' · '+source['id'][:12]):
            st.write(source['text'])
            st.code(source['id'],language=None)
    with st.expander('Structured answer and evidence'):
        st.json(result)

st.title('Enterprise Knowledge Assistant')
st.caption('Evidence-first document answers and revenue analysis · Synthetic Northstar company documents')
st.info('Public portfolio demo: hash-vector + BM25 retrieval, lexical reranking and extractive answers. No paid API or external LLM is used.')
service=get_service()
st.session_state.setdefault('session_id',uuid4().hex)
st.session_state.setdefault('messages',[])
examples={
    'Revenue analysis':'Analyze Q2 revenue and explain why it decreased compared with Q1.',
    'Remote work':'What is the remote work policy?',
    'Security incidents':'How quickly must a security incident be reported?'}
with st.sidebar:
    st.header('Knowledge workspace')
    st.metric('Demo documents',len({c.source for c in service.index.chunks}))
    st.metric('Indexed chunks',len(service.index.chunks))
    st.caption('PDF · Word · CSV · HTML · Markdown')
    st.subheader('Try a question')
    selected=st.selectbox('Example',list(examples))
    run_example=st.button('Ask example',type='primary',use_container_width=True)
    if st.button('Clear conversation'):
        st.session_state.messages=[]
        st.session_state.session_id=uuid4().hex
        st.rerun()
    with st.expander('Included source documents'):
        for path in sorted((ROOT/'data/demo').iterdir()):
            if path.is_file(): st.write(path.name)
    st.caption('Read-only demo corpus. Conversations are separate by browser session; temporary runtime data resets on restart.')

chat,analytics,about=st.tabs(['Ask the assistant','Revenue overview','How it works'])
with chat:
    if not st.session_state.messages:
        st.markdown('Ask a policy question or compare Q1 and Q2 revenue. Each answer includes its supporting document excerpts.')
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message['role']=='user': st.write(message['content'])
            else: display_response(message['response'])
    question=st.chat_input('Ask about a policy or compare Q1 and Q2 revenue',max_chars=2000)
    if run_example: question=examples[selected]
    if question:
        from eki.models import ChatRequest
        validated=ChatRequest(question=question,session_id=st.session_state.session_id)
        with st.spinner('Retrieving evidence and validating citations…'):
            result=service.chat(validated.question,validated.session_id).model_dump()
        st.session_state.messages.extend([{'role':'user','content':question},{'role':'assistant','response':result}])
        # Keep UI history bounded independently of the service's four-question context.
        st.session_state.messages=st.session_state.messages[-20:]
        st.rerun()
with analytics:
    revenue=pd.read_csv(ROOT/'data/demo/revenue.csv')
    st.subheader('Synthetic quarterly revenue')
    st.bar_chart(revenue.set_index('quarter')['revenue'],color='#287d8e')
    st.dataframe(revenue,hide_index=True)
    st.caption('Values are fictional. Ask the revenue comparison example to calculate the change using approved SQL and retrieve the management commentary. Commentary is a reported explanation, not proof of causation.')
with about:
    st.markdown('''### From question to evidence
1. Route policy questions to hybrid retrieval, or revenue comparisons to approved read-only SQL.
2. Combine Qdrant hash-vector matches with BM25 using reciprocal rank fusion.
3. Rerank the candidates and extract supported passages.
4. Return structured claims and validated evidence IDs, or abstain when evidence is insufficient.

This public deployment uses a fixed synthetic corpus. The separate FastAPI platform supports ingestion; public uploads are not enabled here. Hash vectors are a baseline rather than learned semantic embeddings. Extractive answers do not constitute LLM-generated synthesis.
''')
    st.link_button('View source code','https://github.com/Sanchay20Shukla/enterprise-knowledge-intelligence')
