import os
import sys
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.progress import track
from rich.table import Table

from .shared.config import load_config, Config
from .shared.logging_utils import setup_logging, get_logger
from .ingest.github_ingest import GitHubIngestor, Document
from .ingest.confluence_ingest import ConfluenceIngestor, LocalDocsIngestor
from .search.vector_store import VectorStore, KeywordSearch
from .search.retriever import HybridRetriever
from .chat.llm_client import OllamaClient, ChatBot

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', help='Log file path')
@click.pass_context
def main(ctx, config, log_level, log_file):
    """Enterprise Doc Bot - RAG system for GitHub repos and Confluence docs"""
    
    # Setup logging
    log_file_path = Path(log_file) if log_file else None
    setup_logging(level=log_level, log_file=log_file_path)
    
    # Load configuration
    try:
        ctx.obj = load_config(Path(config))
        logger.info(f"Loaded configuration from {config}")
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option('--force', is_flag=True, help='Force re-ingestion of all sources')
@click.pass_context
def ingest(ctx, force):
    """Ingest documents from configured sources"""
    config: Config = ctx.obj
    
    console.print("[bold blue]Starting document ingestion...[/bold blue]")
    
    # Initialize components
    vector_store = VectorStore(
        persist_dir=config.storage.persist_dir,
        model_name="all-MiniLM-L6-v2"
    )
    keyword_search = KeywordSearch()
    
    if force:
        console.print("[yellow]Force flag set - clearing existing index[/yellow]")
        vector_store.clear_index()
        keyword_search.clear_index()
    
    all_documents = []
    
    # Ingest GitHub repositories
    if config.sources.get('git'):
        github_ingestor = GitHubIngestor()
        
        for git_source in track(
            config.sources['git'],
            description="Processing GitHub repositories..."
        ):
            try:
                docs = github_ingestor.ingest_repository(
                    git_source, 
                    chunk_size=config.storage.chunk_size
                )
                all_documents.extend(docs)
                console.print(f"[green]✓[/green] {git_source.name}: {len(docs)} documents")
            except Exception as e:
                console.print(f"[red]✗[/red] {git_source.name}: {e}")
    
    # Ingest Confluence spaces
    if config.sources.get('knowledge_base'):
        confluence_ingestor = ConfluenceIngestor()
        local_ingestor = LocalDocsIngestor()
        
        for kb_source in track(
            config.sources['knowledge_base'],
            description="Processing knowledge base sources..."
        ):
            try:
                if kb_source.type == "confluence":
                    docs = confluence_ingestor.ingest_confluence_space(
                        kb_source,
                        chunk_size=config.storage.chunk_size
                    )
                elif kb_source.type == "local":
                    docs = local_ingestor.ingest_local_docs(
                        kb_source,
                        chunk_size=config.storage.chunk_size
                    )
                else:
                    console.print(f"[yellow]Skipping unknown source type: {kb_source.type}[/yellow]")
                    continue
                
                all_documents.extend(docs)
                console.print(f"[green]✓[/green] {kb_source.name}: {len(docs)} documents")
            except Exception as e:
                console.print(f"[red]✗[/red] {kb_source.name}: {e}")
    
    # Add documents to stores
    if all_documents:
        console.print(f"\n[bold]Adding {len(all_documents)} documents to vector store...[/bold]")
        vector_store.add_documents(all_documents)
        keyword_search.add_documents(all_documents)
        
        console.print("[bold]Saving vector index...[/bold]")
        vector_store.save_index()
        
        # Show stats
        stats = vector_store.get_stats()
        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Documents", str(stats["total_documents"]))
        table.add_row("Index Size", str(stats["index_size"]))
        table.add_row("Embedding Dimension", str(stats["embedding_dimension"]))
        table.add_row("Model", stats["model_name"])
        
        for doc_type, count in stats["document_types"].items():
            table.add_row(f"Documents ({doc_type})", str(count))
        
        console.print(table)
        
    else:
        console.print("[yellow]No documents found to ingest[/yellow]")
    
    console.print("[bold green]Ingestion complete![/bold green]")


@main.command()
@click.option('--host', default='localhost', help='Host to bind the server')
@click.option('--port', default=8501, help='Port to bind the server')
@click.pass_context
def serve(ctx, host, port):
    """Launch the Streamlit web interface"""
    config: Config = ctx.obj
    
    # Check if Ollama is available
    llm_client = OllamaClient(config.llm)
    if not llm_client.is_available():
        console.print(f"[red]Ollama is not available at {config.llm.base_url}[/red]")
        console.print("Please make sure Ollama is running and the model is available.")
        sys.exit(1)
    
    # Check if vector index exists
    index_path = Path(config.storage.persist_dir)
    if not (index_path / "index.faiss").exists():
        console.print("[red]No vector index found. Please run 'ingest' first.[/red]")
        sys.exit(1)
    
    console.print(f"[bold blue]Starting web interface at http://{host}:{port}[/bold blue]")
    
    # Set environment variables for Streamlit app
    os.environ['DOCBOT_CONFIG_PATH'] = 'config.yaml'
    
    # Launch Streamlit
    import subprocess
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        str(Path(__file__).parent / 'web' / 'app.py'),
        '--server.address', host,
        '--server.port', str(port),
        '--server.headless', 'true'
    ])


@main.command()
@click.argument('query', required=True)
@click.option('--k', default=5, help='Number of results to return')
@click.pass_context
def search(ctx, query, k):
    """Search the knowledge base"""
    config: Config = ctx.obj
    
    # Initialize search components
    vector_store = VectorStore(persist_dir=config.storage.persist_dir)
    keyword_search = KeywordSearch()
    
    if not vector_store.documents:
        console.print("[red]No documents in index. Please run 'ingest' first.[/red]")
        sys.exit(1)
    
    # Load documents into keyword search
    keyword_search.add_documents(vector_store.documents)
    
    # Initialize retriever
    retriever = HybridRetriever(vector_store, keyword_search)
    
    # Perform search
    results = retriever.search(query, k=k)
    
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    # Display results
    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Rank", width=4)
    table.add_column("Score", width=8)
    table.add_column("Source", width=30)
    table.add_column("Content Preview", width=60)
    
    for result in results:
        preview = result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content
        preview = preview.replace('\n', ' ')
        
        table.add_row(
            str(result.rank),
            f"{result.score:.3f}",
            result.document.source,
            preview
        )
    
    console.print(table)


@main.command()
@click.option('--interactive', '-i', is_flag=True, help='Interactive chat mode')
@click.pass_context
def chat(ctx, interactive):
    """Chat with the knowledge base"""
    config: Config = ctx.obj
    
    # Initialize components
    vector_store = VectorStore(persist_dir=config.storage.persist_dir)
    keyword_search = KeywordSearch()
    
    if not vector_store.documents:
        console.print("[red]No documents in index. Please run 'ingest' first.[/red]")
        sys.exit(1)
    
    keyword_search.add_documents(vector_store.documents)
    retriever = HybridRetriever(vector_store, keyword_search)
    
    # Initialize LLM
    llm_client = OllamaClient(config.llm)
    if not llm_client.is_available():
        console.print(f"[red]Ollama is not available at {config.llm.base_url}[/red]")
        sys.exit(1)
    
    chatbot = ChatBot(llm_client, config.app.system_prompt)
    
    if interactive:
        console.print("[bold blue]Interactive chat mode. Type 'exit' to quit.[/bold blue]")
        console.print("[dim]Tip: Your questions will be answered using the ingested knowledge base.[/dim]\n")
        
        while True:
            try:
                question = console.input("[bold cyan]You: [/bold cyan]")
                if question.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not question.strip():
                    continue
                
                # Get context from retrieval
                context, search_results = retriever.get_context_for_query(question, max_tokens=4000)
                
                # Generate response
                console.print("[bold green]Bot: [/bold green]", end="")
                response = chatbot.chat(question, context=context)
                console.print(response.content)
                console.print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    else:
        console.print("[yellow]Use --interactive flag for chat mode[/yellow]")


@main.command()
@click.pass_context
def status(ctx):
    """Show system status and statistics"""
    config: Config = ctx.obj
    
    # Check vector store
    vector_store = VectorStore(persist_dir=config.storage.persist_dir)
    stats = vector_store.get_stats()
    
    # Check Ollama
    llm_client = OllamaClient(config.llm)
    ollama_status = "✓ Available" if llm_client.is_available() else "✗ Not Available"
    
    # Create status table
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Vector Store", f"✓ {stats['total_documents']} documents")
    table.add_row("Ollama LLM", ollama_status)
    table.add_row("Model", config.llm.model)
    table.add_row("Embedding Model", stats["model_name"])
    
    console.print(table)
    
    if stats["sources"]:
        sources_table = Table(title="Ingested Sources")
        sources_table.add_column("Source", style="cyan")
        sources_table.add_column("Documents", style="green")
        
        for source, count in stats["sources"].items():
            sources_table.add_row(source, str(count))
        
        console.print(sources_table)


if __name__ == '__main__':
    main()