# RAG Dialogue System Demo

## Features
- Vector-Database Integrated
- Retrieval-Augmented Generation
- Multi-Agent Architecture (Knowledge Reader + Content Analyst)
- Predefined Tools & Test Queries
- Clear Workflow Demonstration

## How to use

### Setting your OpenAI API key

Set the `QWEN_API_KEY` environment variable in the `.env` file at the root folder. `Python-dotenv` is needed to properly load the environment variables from the `.env` file.

### Install dependencies

(Upon setting up the virtual environment) Install the dependencies by running the following commands at the root folder:

```bash
pip install -r requirements.txt
```

### Run the app

1. Run the main script to start the system:

```bash
python .\code\role_play_rag.py
```

2. Upon loading the data from `.files\small_ocr_content_list.json`, the system would prompt you with below modes:

```
a) Enter your query directly 
b) Input 'test' for testing queries
c) Input 'exit' to quit the system
```
