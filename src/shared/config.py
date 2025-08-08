import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class SwaggerSource:
    url: str
    name: str

@dataclass
class GitSource:
    url: str
    name: str
    branch: str = "main"

@dataclass
class KnowledgeBaseSource:
    name: str
    type: str  # "confluence", "local", etc.
    url: Optional[str] = None
    path: Optional[str] = None

@dataclass
class LLMConfig:
    model: str
    request_timeout: int = 120
    base_url: str = "http://localhost:11434"

@dataclass
class StorageConfig:
    persist_dir: str
    chunk_size: int = 1000

@dataclass
class AppConfig:
    title: str
    system_prompt: str

@dataclass
class Config:
    sources: Dict[str, List[Any]]
    llm: LLMConfig
    storage: StorageConfig
    app: AppConfig
    
    @classmethod
    def load_from_file(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse sources
        sources = {}
        if 'swagger' in data.get('sources', {}):
            sources['swagger'] = [SwaggerSource(**item) for item in data['sources']['swagger']]
        if 'git' in data.get('sources', {}):
            sources['git'] = [GitSource(**item) for item in data['sources']['git']]
        if 'knowledge_base' in data.get('sources', {}):
            sources['knowledge_base'] = [KnowledgeBaseSource(**item) for item in data['sources']['knowledge_base']]
        
        return cls(
            sources=sources,
            llm=LLMConfig(**data['llm']),
            storage=StorageConfig(**data['storage']),
            app=AppConfig(**data['app'])
        )


def load_config(config_path: Path) -> Config:
    """Load configuration from file path."""
    return Config.load_from_file(str(config_path))