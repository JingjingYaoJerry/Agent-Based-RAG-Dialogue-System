## Agent-Based RAG Dialogue System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CAMEL-AI](https://img.shields.io/badge/CAMEL--AI-0.2+-orange.svg)](https://docs.camel-ai.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-client-yellowgreen.svg)](https://qdrant.tech/)

An example Retrieval-Augmented Generation (RAG) dialogue demo built with CAMEL-AI. The project ingests preprocessed OCR text slices
(`files/small_ocr_content_list.json`) into a Qdrant-backed vector store, performs semantic retrieval, and uses two CAMEL agents
(ContentAnalyst & KnowledgeReader) to produce concise, evidence-backed answers.

---

### Key Features

* Vector database ingestion and storage via `code/qdrant.py` (uses SentenceTransformerEncoder for embeddings).
* Semantic retrieval wrapper `code/vector_retriever.py` (returns `file_name` and `content`).
* RAG orchestration in `code/role_play_rag.py` using CAMEL-AI RolePlaying society.
* Command-line demo with a `test` mode that runs the five assignment test questions and prints retrieved snippets + generated answers.
* Small FunctionTools (preprocessing / answer-type detection / display) wired into the agents for demonstration.

---

### Setup and Installation (PowerShell)

1. Clone the repository:

```powershell
git clone https://github.com/JingjingYaoJerry/Agent-Based-RAG-Dialogue-System.git
cd Agent-Based-RAG-Dialogue-System
```

2. (Optional but recommended) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Create a `.env` file at the repository root and add your model/API key(s):

```
QWEN_API_KEY=your_api_key_here
```

---

### Run the demo (CLI)

Start the main script and follow prompts:

```powershell
python .\code\role_play_rag.py
```

Interaction options:

- Type a query and press Enter → the system will display the top-k retrieved snippets and then the agent-generated answer.
- Type `test` → runs the five assignment test questions and prints retrieval + answers.
- Type `exit` → quit the program.

---

### Project structure

```
Agent-Based-RAG-Dialogue-System/
├── code/
│   ├── qdrant.py                # Qdrant DB helper (ingest/save_text)
│   ├── vector_retriever.py      # Wrapper around CAMEL VectorRetriever
│   ├── role_play_rag.py         # Main RAG orchestration and CLI demo
│   └── tests/                   # Lightweight simulation tests
├── files/
│   └── small_ocr_content_list.json
├── requirements.txt
├── README.md
└── assignment_1.pdf
```

---

### Acknowledgements

This project is built using CAMEL-AI (https://docs.camel-ai.org/) and the Qdrant vector store. The dataset slices come from the provided `small_ocr_content_list.json`.