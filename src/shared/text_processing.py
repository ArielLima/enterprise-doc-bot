import re
from typing import List, Dict, Any
from pathlib import Path


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    return text


def chunk_text(
    text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200,
    preserve_sentences: bool = True
) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        if preserve_sentences:
            sentence_end = text.rfind('.', start, end)
            if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunks.append(text[start:end])
        start = end - chunk_overlap
    
    return chunks


def extract_code_blocks(content: str) -> List[Dict[str, Any]]:
    code_pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(code_pattern, content, re.DOTALL)
    
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            'language': language or 'text',
            'code': code.strip(),
            'type': 'code_block'
        })
    
    return code_blocks


def get_file_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    
    code_extensions = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.rs': 'rust',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
    }
    
    doc_extensions = {
        '.md': 'markdown',
        '.txt': 'text',
        '.rst': 'restructuredtext',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
    }
    
    if suffix in code_extensions:
        return code_extensions[suffix]
    elif suffix in doc_extensions:
        return doc_extensions[suffix]
    else:
        return 'text'


def should_skip_file(file_path: Path) -> bool:
    skip_patterns = [
        r'\.git/',
        r'node_modules/',
        r'__pycache__/',
        r'\.pyc$',
        r'\.pyo$',
        r'\.o$',
        r'\.so$',
        r'\.dylib$',
        r'\.exe$',
        r'\.bin$',
        r'\.jpg$',
        r'\.jpeg$',
        r'\.png$',
        r'\.gif$',
        r'\.pdf$',
        r'\.zip$',
        r'\.tar\.gz$',
    ]
    
    file_str = str(file_path)
    return any(re.search(pattern, file_str, re.IGNORECASE) for pattern in skip_patterns)