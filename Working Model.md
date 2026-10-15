# Working Model — Enterprise Knowledge Assistant

**[Open the live Streamlit website](https://sanchay-enterprise-knowledge-assistant.streamlit.app/)**

Reviewers can open the app directly without installing software, signing in or providing an API key. If the app has been put to sleep by hosting, allow it to wake up.

## Try it

1. In the sidebar, select **Revenue analysis** and click **Ask example**.
2. Review the calculated Q1-to-Q2 change, source excerpts and cited evidence IDs.
3. Ask **What is the remote work policy?** in chat.
4. Expand the supporting PDF excerpt and inspect its page reference.
5. Explore the **Revenue overview** and **How it works** tabs.

## What runs behind the interface

The app invokes the existing LangGraph workflow: route the question, run an approved SQL comparison when needed, calculate the revenue change, retrieve documents with Qdrant hash vectors plus BM25, rerank and return extractive claims with validated citation IDs. Answers can abstain or be blocked when the request is unsupported.

All company documents and financial figures are synthetic. The example revenue changes from **1,200,000 to 960,000**, a **240,000 decline (20%)**. Management commentary provides reported explanations rather than proven causal evidence. Hash vectors are a retrieval baseline rather than learned semantic embeddings; the public app does not call an external LLM.

## Verification and source

- Local suite: **26 passed, one external-service test skipped**.
- Streamlit runtime checks: revenue answer and citations; policy chat and PDF source citation.
- [Streamlit app](streamlit_app.py)
- [Service](eki/service.py)
- [LangGraph workflow](eki/workflow.py)
- [Retrieval](eki/retrieval.py)
- [Approved SQL](eki/sql_agent.py)
- [Synthetic source documents](data/demo)
- [Saved evaluation results](eval/results.json)

The public corpus is read-only and is shared across visitors. Conversation IDs are separate per browser session; temporary indexes and state reset on restart. The FastAPI implementation remains available for local ingestion and development.
