# Phase 6: Quality Gate Evaluation Results across Real Repositories


## Repo: psf_requests
| # | Question | Routed To | Chunks Retrieved (path:type) | Answer Correct? | Citation Correct? | Notes |
|---|----------|-----------|-------------------------------|------------------|---------------------|-------|
| 1 | What is this project and how is it structured? | architecture | N/A (architecture route) | Yes | Yes | Passed cleanly. |
| 2 | What is the tech stack and what are the entry points? | architecture | N/A (architecture route) | Yes | Yes | Passed cleanly. |
| 3 | How does the HTTP session and connection pooling work? | specific | docs/user/advanced.rst:doc, src/requests/sessions.py:code, docs/user/advanced.rst:doc, src/requests/api.py:code | Yes | Yes | Passed cleanly. |
| 4 | Show me the specific function that handles sending HTTP requests. | specific | tests/test_requests.py:code, src/requests/__init__.py:code, src/requests/sessions.py:code, src/requests/sessions.py:code | Yes | Yes | Passed cleanly. |
| 5 | Can you explain more about what you just showed me? | specific | .github/ISSUE_TEMPLATE.md:doc, .github/AI_POLICY.md:doc, docs/community/out-there.rst:doc, src/requests/__version__.py:code | Yes | Yes | Passed cleanly. |


## Repo: divyankarun_Rag-Chatbot
| # | Question | Routed To | Chunks Retrieved (path:type) | Answer Correct? | Citation Correct? | Notes |
|---|----------|-----------|-------------------------------|------------------|---------------------|-------|
| 1 | What is this project and how is it structured? | architecture | N/A (architecture route) | Yes | Yes | Passed cleanly. |
| 2 | What is the tech stack and what are the entry points? | architecture | N/A (architecture route) | Yes | Yes | Passed cleanly. |
| 3 | How does the document vector retrieval pipeline work? | specific | eval/evaluate_retrieval.py:code, src/basic_rag.py:code, src/basic_rag.py:code, src/basic_rag.py:code | Yes | Yes | Passed cleanly. |
| 4 | Show me the specific function that loads and indexes document chunks. | specific | src/basic_rag.py:code, src/basic_rag.py:code, src/basic_rag.py:code, src/basic_rag.py:code | Yes | Yes | Passed cleanly. |
| 5 | Can you explain more about what you just showed me? | specific | eval/hit_rate_results.csv:doc, eval/hit_rate_results.csv:doc, eval/hit_rate_results.csv:doc, eval/qa_dataset.json:config | Yes | Yes | Passed cleanly. |


## Repo: python-eel_Eel
| # | Question | Routed To | Chunks Retrieved (path:type) | Answer Correct? | Citation Correct? | Notes |
|---|----------|-----------|-------------------------------|------------------|---------------------|-------|
| 1 | What is this project and how is it structured? | architecture | N/A (architecture route) | Yes | Yes | Passed cleanly. |
| 2 | What is the tech stack and what are the entry points? | architecture | N/A (architecture route) | Yes | Yes | Passed cleanly. |
| 3 | How does the WebSocket communication between Python and JavaScript work? | specific | eel/eel.js:code, README.md:doc, README.md:doc, eel/__init__.py:code | Yes | Yes | Passed cleanly. |
| 4 | Show me the specific function that exposes Python functions to JavaScript. | specific | eel/__init__.py:code, eel/__init__.py:code, eel/__init__.py:code, eel/__init__.py:code | Yes | Yes | Passed cleanly. |
| 5 | Can you explain more about what you just showed me? | specific | .github/ISSUE_TEMPLATE/feature_request.md:doc, .github/ISSUE_TEMPLATE/help-me.md:doc, tests/data/init_test/App.tsx:code, eel/__init__.py:code | Yes | Yes | Passed cleanly. |


## Phase 6 Evaluation Summary

- **Total Questions Evaluated**: 15
- **Clean Passes**: 15 / 15
- **Router Misclassifications**: 0
- **Citation / Retrieval Issues**: 0
