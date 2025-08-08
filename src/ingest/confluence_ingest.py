import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urljoin

from atlassian import Confluence
import requests
from bs4 import BeautifulSoup

from ..shared.config import KnowledgeBaseSource
from ..shared.logging_utils import get_logger
from ..shared.text_processing import chunk_text, clean_text

logger = get_logger(__name__)


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    source: str
    doc_type: str


class ConfluenceIngestor:
    def __init__(self, username: Optional[str] = None, token: Optional[str] = None):
        self.username = username or os.getenv("CONFLUENCE_USERNAME")
        self.token = token or os.getenv("CONFLUENCE_TOKEN")
        self.confluence = None
    
    def _init_confluence_client(self, base_url: str) -> None:
        if not self.username or not self.token:
            raise ValueError("Confluence username and token are required")
        
        self.confluence = Confluence(
            url=base_url,
            username=self.username,
            password=self.token,
            cloud=True
        )
    
    def ingest_confluence_space(self, kb_source: KnowledgeBaseSource, chunk_size: int = 1000) -> List[Document]:
        logger.info(f"Ingesting Confluence space: {kb_source.name}")
        
        if not kb_source.url:
            logger.error("Confluence URL is required")
            return []
        
        self._init_confluence_client(kb_source.url)
        documents = []
        
        try:
            # Extract space key from URL or use provided space key
            space_key = self._extract_space_key(kb_source.url)
            if not space_key:
                logger.error("Could not extract space key from URL")
                return []
            
            logger.info(f"Fetching pages from space: {space_key}")
            pages = self.confluence.get_all_pages_from_space(
                space=space_key,
                start=0,
                limit=500,
                expand="body.storage,metadata.labels,version,ancestors"
            )
            
            for page in pages:
                try:
                    page_docs = self._process_page(page, kb_source, space_key, chunk_size)
                    documents.extend(page_docs)
                except Exception as e:
                    logger.warning(f"Failed to process page {page.get('id', 'unknown')}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to fetch Confluence pages: {e}")
            return documents
        
        logger.info(f"Extracted {len(documents)} documents from {kb_source.name}")
        return documents
    
    def _extract_space_key(self, url: str) -> Optional[str]:
        # Try to extract space key from various Confluence URL formats
        patterns = [
            r'/spaces/([A-Z0-9]+)/',
            r'/display/([A-Z0-9]+)/',
            r'spaceKey=([A-Z0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _process_page(self, page: Dict[str, Any], kb_source: KnowledgeBaseSource, space_key: str, chunk_size: int) -> List[Document]:
        documents = []
        
        page_id = page['id']
        title = page['title']
        
        # Extract content from storage format
        body_content = ""
        if 'body' in page and 'storage' in page['body']:
            body_content = page['body']['storage']['value']
        
        # Convert HTML to clean text
        clean_content = self._html_to_text(body_content)
        
        if not clean_content.strip():
            return documents
        
        # Add title and context
        full_content = f"# {title}\n\n{clean_content}"
        
        # Get page metadata
        labels = []
        if 'metadata' in page and 'labels' in page['metadata']:
            labels = [label['name'] for label in page['metadata']['labels']['results']]
        
        # Get ancestors for breadcrumb
        ancestors = []
        if 'ancestors' in page:
            ancestors = [ancestor['title'] for ancestor in page['ancestors']]
        
        # Chunk the content
        chunks = chunk_text(full_content, chunk_size, chunk_size // 5)
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                content=chunk,
                metadata={
                    "source": kb_source.name,
                    "confluence_url": kb_source.url,
                    "space_key": space_key,
                    "page_id": page_id,
                    "page_title": title,
                    "labels": labels,
                    "ancestors": ancestors,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "last_modified": page.get('version', {}).get('when', ''),
                    "page_url": f"{kb_source.url}/pages/viewpage.action?pageId={page_id}"
                },
                source=f"{kb_source.name}:{title}",
                doc_type="documentation"
            )
            documents.append(doc)
        
        return documents
    
    def _html_to_text(self, html_content: str) -> str:
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
        
        # Handle code blocks
        code_blocks = soup.find_all(['pre', 'code'])
        for code in code_blocks:
            code.string = f"\n```\n{code.get_text()}\n```\n"
        
        # Handle tables
        tables = soup.find_all('table')
        for table in tables:
            table_text = self._table_to_text(table)
            table.replace_with(table_text)
        
        # Get text and clean it
        text = soup.get_text()
        return clean_text(text)
    
    def _table_to_text(self, table) -> str:
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                cells.append(td.get_text().strip())
            if cells:
                rows.append(" | ".join(cells))
        
        return "\n".join(rows) + "\n"


class LocalDocsIngestor:
    def ingest_local_docs(self, kb_source: KnowledgeBaseSource, chunk_size: int = 1000) -> List[Document]:
        logger.info(f"Ingesting local documentation: {kb_source.name}")
        
        if not kb_source.path:
            logger.error("Local documentation path is required")
            return []
        
        from pathlib import Path
        from ..shared.text_processing import should_skip_file, get_file_type
        
        documents = []
        docs_path = Path(kb_source.path)
        
        if not docs_path.exists():
            logger.error(f"Documentation path does not exist: {docs_path}")
            return []
        
        for file_path in docs_path.rglob("*"):
            if file_path.is_file() and not should_skip_file(file_path):
                try:
                    relative_path = file_path.relative_to(docs_path)
                    content = self._read_file_content(file_path)
                    
                    if not content.strip():
                        continue
                    
                    file_type = get_file_type(file_path)
                    clean_content = clean_text(content)
                    
                    chunks = chunk_text(clean_content, chunk_size, chunk_size // 5)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            content=chunk,
                            metadata={
                                "source": kb_source.name,
                                "file_path": str(relative_path),
                                "file_type": file_type,
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            },
                            source=f"{kb_source.name}:{relative_path}",
                            doc_type="documentation"
                        )
                        documents.append(doc)
                
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    continue
        
        logger.info(f"Extracted {len(documents)} documents from {kb_source.name}")
        return documents
    
    def _read_file_content(self, file_path) -> str:
        encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        logger.warning(f"Could not decode file {file_path}")
        return ""