"""
LLM abstraction layer for easy switching between different models.
Supports both local and remote models.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from chat messages"""
        pass


class OllamaLLM(BaseLLM):
    """Ollama local model implementation"""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._import_ollama()
    
    def _import_ollama(self):
        """Import ollama with error handling"""
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            raise ImportError("Ollama not installed. Install with: pip install ollama")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama"""
        try:
            response = self.ollama.generate(
                model=self.model_name,
                prompt=prompt,
                **kwargs
            )
            return response['response']
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat response using Ollama"""
        try:
            response = self.ollama.chat(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Ollama chat failed: {e}")


class TransformersLLM(BaseLLM):
    """Hugging Face Transformers local model implementation"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """Load the transformers model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            raise ImportError("Transformers not installed. Install with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """Generate text using Transformers"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            raise RuntimeError(f"Transformers generation failed: {e}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat response using Transformers"""
        # Convert messages to prompt format
        prompt = self._format_messages(messages)
        return self.generate(prompt, **kwargs)
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt"""
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"Human: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted) + "\nAssistant:"


class OpenAILLM(BaseLLM):
    """OpenAI API implementation"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self._setup_client()
    
    def _setup_client(self):
        """Setup OpenAI client"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI not installed. Install with: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI"""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                **kwargs
            )
            return response.choices[0].text
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI chat failed: {e}")


class LLMFactory:
    """Factory class for creating LLM instances"""
    
    @staticmethod
    def create_llm(config: Dict[str, Any]) -> BaseLLM:
        """Create LLM instance based on configuration"""
        llm_type = config.get("type", "ollama").lower()
        
        if llm_type == "ollama":
            return OllamaLLM(
                model_name=config.get("model_name", "llama2"),
                base_url=config.get("base_url", "http://localhost:11434")
            )
        elif llm_type == "transformers":
            return TransformersLLM(
                model_name=config.get("model_name", "microsoft/DialoGPT-medium"),
                device=config.get("device", "auto")
            )
        elif llm_type == "openai":
            return OpenAILLM(
                model_name=config.get("model_name", "gpt-3.5-turbo"),
                api_key=config.get("api_key")
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")


def load_llm_config(config_path: str = "llm_config.json") -> Dict[str, Any]:
    """Load LLM configuration from file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return default configuration
        return {
            "type": "ollama",
            "model_name": "llama2",
            "base_url": "http://localhost:11434"
        }


def save_llm_config(config: Dict[str, Any], config_path: str = "llm_config.json"):
    """Save LLM configuration to file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
