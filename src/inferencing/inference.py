from abc import ABC, abstractmethod
from typing import Dict, Any
import requests
import json
import os

# Constants
PLACEHOLDER_ENGINE = "placeholder"
PLACEHOLDER_MODEL = "placeholder-model"

class InferenceEngine(ABC):
    """Abstract base class for inference engines"""
    
    @abstractmethod
    def generate(self, prompt: str, params: Dict[Any, Any]) -> str:
        """Generate a response from the model"""
        pass

    @staticmethod
    def create_engine(engine_type: str = "ollama", model_name: str = "huihui_ai/llama3.2-abliterated") -> 'InferenceEngine':
        """Factory method to create appropriate inference engine"""
        if engine_type == "ollama":
            return OllamaEngine(model_name)
        elif engine_type == "claude":
            return ClaudeEngine(model_name)
        elif engine_type == PLACEHOLDER_ENGINE:
            return PlaceholderEngine(model_name)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")

class OllamaEngine(InferenceEngine):
    """Ollama inference engine implementation"""
    
    def __init__(self, model_name: str = "huihui_ai/llama3.2-abliterated"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        self.generate_url = f"{self.base_url}/generate"
        self._check_connection()
        self._check_model()
    
    def _check_connection(self):
        """Test connection to Ollama server"""
        try:
            requests.get(self.base_url)
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama server: {str(e)}")

    def _check_model(self):
        """Verify if the specified model is available"""
        try:
            response = requests.get(f"{self.base_url}/tags")
            available_models = [model['name'] for model in response.json()['models']]
            if self.model_name not in available_models:
                raise ValueError(f"Model {self.model_name} not found. Available models: {', '.join(available_models)}")
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot check available models: {str(e)}")
    
    def generate(self, prompt: str, params: Dict[Any, Any]) -> str:
        """Generate a response from the model"""
        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": params.get('creativity', 0.5),
                    "stream": False,
                    "context": params.get('context', []),
                    "system": params.get('system_prompt', ''),
                    "format": params.get('format', 'text')
                }
            )
            response.raise_for_status()
            return response.json()['response']
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return f"Model {self.model_name} not found"
            elif e.response.status_code == 500:
                return "Internal Ollama server error"
            else:
                return f"HTTP error occurred: {str(e)}"
        except (KeyError, json.JSONDecodeError) as e:
            return f"Error parsing Ollama response: {str(e)}"
        except requests.RequestException as e:
            return f"Network error with Ollama inference: {str(e)}"

class PlaceholderEngine(InferenceEngine):
    """Placeholder engine for testing and development"""
    
    def __init__(self, model_name: str = PLACEHOLDER_MODEL):
        self.model_name = model_name
    
    def generate(self, prompt: str, params: Dict[Any, Any]) -> str:
        """Return a placeholder response"""
        return f"[Placeholder Response]\nModel: {self.model_name}\nPrompt: {prompt}\nParams: {params}"

class ClaudeEngine(InferenceEngine):
    """Claude API inference engine implementation"""
    
    def __init__(self, model_name: str = "claude-2"):
        self.model_name = model_name
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    def generate(self, prompt: str, params: Dict[Any, Any]) -> str:
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": params.get('creativity', 0.5)
                }
            )
            return response.json()['content'][0]['text']
        except (KeyError, json.JSONDecodeError, requests.RequestException) as e:
            return f"Error with Claude inference: {str(e)}"
