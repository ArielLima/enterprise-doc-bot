import json
import time
from typing import Dict, Any, Optional, List, Generator
import requests
from dataclasses import dataclass

from ..shared.config import LLMConfig
from ..shared.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ChatMessage:
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ChatResponse:
    content: str
    model: str
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip('/')
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        logger.info(f"Pulling model: {model_name}")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if "status" in data:
                            logger.info(f"Pull status: {data['status']}")
                        if data.get("status") == "success":
                            logger.info(f"Successfully pulled {model_name}")
                            return True
                    except json.JSONDecodeError:
                        continue
            
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> ChatResponse:
        if stream:
            return self._chat_stream(messages, temperature, max_tokens)
        else:
            return self._chat_complete(messages, temperature, max_tokens)
    
    def _chat_complete(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: Optional[int]
    ) -> ChatResponse:
        payload = {
            "model": self.config.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            return ChatResponse(
                content=data["message"]["content"],
                model=data.get("model", self.config.model),
                total_duration=data.get("total_duration"),
                eval_count=data.get("eval_count"),
                eval_duration=data.get("eval_duration")
            )
            
        except requests.RequestException as e:
            logger.error(f"Chat request failed: {e}")
            raise
    
    def _chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: Optional[int]
    ) -> Generator[str, None, None]:
        payload = {
            "model": self.config.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.request_timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except requests.RequestException as e:
            logger.error(f"Streaming chat request failed: {e}")
            raise


class ChatBot:
    def __init__(self, llm_client: OllamaClient, system_prompt: str):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.conversation_history: List[ChatMessage] = []
    
    def chat(
        self,
        user_message: str,
        context: str = "",
        stream: bool = False,
        clear_history: bool = False
    ) -> ChatResponse:
        if clear_history:
            self.conversation_history = []
        
        # Build the prompt with context
        if context:
            enhanced_prompt = f"""Based on the following context, please answer the user's question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {user_message}"""
        else:
            enhanced_prompt = user_message
        
        # Prepare messages
        messages = [ChatMessage(role="system", content=self.system_prompt)]
        messages.extend(self.conversation_history)
        messages.append(ChatMessage(role="user", content=enhanced_prompt))
        
        # Get response
        response = self.llm_client.chat(messages, stream=stream, temperature=0.1)
        
        # Update conversation history
        self.conversation_history.append(ChatMessage(role="user", content=user_message))
        self.conversation_history.append(ChatMessage(role="assistant", content=response.content))
        
        # Keep history manageable (last 10 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def clear_history(self):
        self.conversation_history = []
    
    def get_history(self) -> List[ChatMessage]:
        return self.conversation_history.copy()