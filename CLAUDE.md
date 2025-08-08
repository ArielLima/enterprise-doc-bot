# Enterprise Doc Bot - Context for Claude

## Project Overview
A self-hosted RAG system that ingests GitHub repositories and Confluence docs to answer questions about codebases and documentation. Uses local LLMs via Ollama for privacy.

## Architecture
```
src/
├── shared/          # Common utilities, config, logging
├── ingest/          # Data ingestion from various sources
├── search/          # Vector + keyword hybrid search
└── chat/            # LLM interface and response generation
```

## Current Configuration (config.yaml)
- **LLM**: mistral:7b-instruct-q4_0 via Ollama
- **Vector DB**: FAISS with local persistence
- **Sources**: GitHub repos, Swagger/OpenAPI, Confluence, local docs
- **Interface**: Streamlit UI planned

## Key Dependencies Needed
- requests, PyGithub (GitHub API)
- atlassian-python-api (Confluence)
- sentence-transformers (embeddings)
- faiss-cpu (vector search)
- ollama (local LLM)
- streamlit (web UI)
- pydantic (config validation)

## Coding Conventions
- Type hints throughout
- Pydantic for config/data models
- Structured logging
- Clean separation of concerns
- Error handling with retries for API calls

## Implementation Strategy
1. Build shared utilities first (config, logging, text processing)
2. Implement ingestion modules (GitHub, Confluence)
3. Set up vector database and search
4. Create query interface and LLM integration
5. Add CLI and web interface

## Testing Approach
- Unit tests for core logic
- Integration tests with mock APIs
- CLI commands: `npm run lint`, `npm run typecheck` (check if these exist)