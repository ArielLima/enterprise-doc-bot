import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import git
from github import Github, Auth
from github.Repository import Repository

from ..shared.config import GitSource
from ..shared.logging_utils import get_logger
from ..shared.text_processing import chunk_text, get_file_type, should_skip_file, clean_text

logger = get_logger(__name__)


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    source: str
    doc_type: str


class GitHubIngestor:
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        if self.github_token:
            auth = Auth.Token(self.github_token)
            self.github_client = Github(auth=auth)
        else:
            self.github_client = Github()
            logger.warning("No GitHub token provided. Rate limits will be lower.")
    
    def ingest_repository(self, git_source: GitSource, chunk_size: int = 1000) -> List[Document]:
        logger.info(f"Ingesting GitHub repository: {git_source.name}")
        
        documents = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / git_source.name
            
            try:
                logger.info(f"Cloning {git_source.url} to {repo_path}")
                repo = git.Repo.clone_from(
                    git_source.url,
                    repo_path,
                    branch=git_source.branch,
                    depth=1
                )
                
                documents.extend(self._process_repository(repo_path, git_source, chunk_size))
                
                if self.github_token:
                    documents.extend(self._get_repo_metadata(git_source))
                
            except git.GitCommandError as e:
                logger.error(f"Failed to clone repository {git_source.url}: {e}")
                return documents
            except Exception as e:
                logger.error(f"Error processing repository {git_source.name}: {e}")
                return documents
        
        logger.info(f"Extracted {len(documents)} documents from {git_source.name}")
        return documents
    
    def _process_repository(self, repo_path: Path, git_source: GitSource, chunk_size: int) -> List[Document]:
        documents = []
        
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and not should_skip_file(file_path):
                try:
                    relative_path = file_path.relative_to(repo_path)
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
                                "source": git_source.name,
                                "repository_url": git_source.url,
                                "branch": git_source.branch,
                                "file_path": str(relative_path),
                                "file_type": file_type,
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            },
                            source=f"{git_source.name}:{relative_path}",
                            doc_type="code" if file_type in ["python", "javascript", "typescript", "java", "cpp", "c", "rust", "go"] else "document"
                        )
                        documents.append(doc)
                
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    continue
        
        return documents
    
    def _read_file_content(self, file_path: Path) -> str:
        encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        logger.warning(f"Could not decode file {file_path}")
        return ""
    
    def _get_repo_metadata(self, git_source: GitSource) -> List[Document]:
        try:
            repo_name = git_source.url.split('/')[-1].replace('.git', '')
            owner_name = git_source.url.split('/')[-2].split('@')[-1].split(':')[-1]
            
            repo: Repository = self.github_client.get_repo(f"{owner_name}/{repo_name}")
            
            metadata_doc = Document(
                content=f"""# {repo.name}

{repo.description or 'No description available'}

**Language:** {repo.language}
**Stars:** {repo.stargazers_count}
**Forks:** {repo.forks_count}
**Last Updated:** {repo.updated_at}

## README

{repo.get_readme().decoded_content.decode('utf-8') if repo.get_readme() else 'No README available'}
""",
                metadata={
                    "source": git_source.name,
                    "repository_url": git_source.url,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "language": repo.language,
                    "updated_at": str(repo.updated_at)
                },
                source=f"{git_source.name}:metadata",
                doc_type="metadata"
            )
            
            return [metadata_doc]
            
        except Exception as e:
            logger.warning(f"Failed to get repository metadata: {e}")
            return []