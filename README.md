# Enterprise Doc Bot

A self-hosted Retrieval-Augmented-Generation (RAG) assistant that ingests **GitHub repositories** and **Confluence docs** to answer questions about codebases and documentation using local LLMs.

## ✨ Features
- **Multi-source ingestion** – GitHub repos, Confluence spaces, local docs
- **Vector + keyword hybrid search** for accurate retrieval
- **CPU-friendly runtime** – runs locally using quantized models via Ollama
- **Streamlit web UI** – chat interface with source citations
- **CLI interface** – command-line tools for ingestion and chat
- **Privacy-first** – everything runs locally, no external API calls

## 🚀 Quick start

### 1. Install Prerequisites
```bash
# Install Ollama (local LLM server)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service (required before pulling models)
ollama serve &

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Setup LLM Model
```bash
# Pull a quantized model (this may take a few minutes)
ollama pull mistral:7b-instruct-q4_0

# Verify model is available
ollama list
```

### 3. Configure Sources
Edit `config.yaml` to add your GitHub repos and Confluence spaces:

```yaml
sources:
  git:
    - url: "https://github.com/your-org/your-repo.git"
      name: "Your Project"
      branch: "main"
  
  knowledge_base:
    - url: "https://your-company.atlassian.net/wiki"
      type: "confluence"
      name: "Company Docs"
    - path: "./docs"
      type: "local"
      name: "Local Documentation"
```

### 4. Ingest & Serve
```bash
# Ingest documents (this may take several minutes)
python main.py ingest

# Launch web interface at http://localhost:8501
python main.py serve
```

### 5. Alternative: CLI Chat
```bash
# Interactive command-line chat
python main.py chat --interactive

# One-time search
python main.py search "How do I configure the API?"

# Check system status
python main.py status
```

## 🔧 Environment Variables
For private repositories and Confluence, set these:
```bash
export GITHUB_TOKEN="your_github_token"
export CONFLUENCE_USERNAME="your_email@company.com"
export CONFLUENCE_TOKEN="your_confluence_token"
```

## 📊 Commands Reference
- `python main.py ingest` - Process all configured sources
- `python main.py ingest --force` - Re-process everything (clears index)
- `python main.py serve` - Start web interface
- `python main.py chat -i` - Interactive CLI chat
- `python main.py search "query"` - Search knowledge base
- `python main.py status` - Show system statistics

## 🐛 Troubleshooting
- **"Ollama not available"**: Run `ollama serve` first
- **"No documents found"**: Run `python main.py ingest` first
- **Import errors**: Activate virtual environment with `source venv/bin/activate`
- **Rate limits**: Set `GITHUB_TOKEN` for higher GitHub API limits
- **Memory issues**: Use smaller models like `mistral:7b-instruct-q4_0`

## 📁 Project Structure
```
enterprise-doc-bot/
├── src/
│   ├── shared/          # Config, logging, text processing
│   ├── ingest/          # GitHub & Confluence ingestion
│   ├── search/          # Vector store & hybrid search
│   ├── chat/            # LLM client & chat interface
│   └── web/             # Streamlit web UI
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
└── main.py             # CLI entry point
```