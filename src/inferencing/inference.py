from abc import ABC, abstractmethod
from typing import Dict, Any
import requests
import json
import os

class InferenceEngine(ABC):
    """Abstract base class for inference engines"""
    
    @abstractmethod
    def generate(self, prompt: str, params: Dict[Any, Any]) -> str:
        """Generate a response from the model"""
        pass

    @staticmethod
    def create_engine(engine_type: str = "ollama", model_name: str = "llama3") -> 'InferenceEngine':
        """Factory method to create appropriate inference engine"""
        if engine_type == "ollama":
            return OllamaEngine(model_name)
        elif engine_type == "claude":
            return ClaudeEngine(model_name)
        elif engine_type == "openai":
            return OpenAIEngine(model_name)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
    
    @staticmethod
    def get_available_engines() -> list:
        """Return list of available inference engines"""
        return ["ollama", "claude", "openai"]

class OllamaEngine(InferenceEngine):
    """Ollama inference engine implementation"""
    
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        self.generate_url = f"{self.base_url}/generate"
        try:
            self._check_connection()
            self._check_model()
        except Exception as e:
            print(f"Warning: Ollama initialization issue: {str(e)}")
            # Continue anyway - the model might be available later
    
    @staticmethod
    def get_available_models() -> list:
        """Get list of available Ollama models"""
        try:
            base_url = "http://localhost:11434/api"
            response = requests.get(f"{base_url}/tags", timeout=5)
            if response.status_code == 200:
                return [model['name'] for model in response.json().get('models', [])]
            return []
        except Exception as e:
            print(f"Warning: Cannot fetch Ollama models: {str(e)}")
            return []
    
    def _check_connection(self):
        """Test connection to Ollama server"""
        try:
            requests.get(self.base_url, timeout=2)  # Short timeout to avoid hanging
        except requests.RequestException as e:
            print(f"Warning: Cannot connect to Ollama server: {str(e)}")
            raise ConnectionError(f"Cannot connect to Ollama server: {str(e)}")

    def _check_model(self):
        """Verify if the specified model is available"""
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=5)  # Longer timeout
            available_models = [model['name'] for model in response.json()['models']]
            
            # Check if the exact model name exists
            if self.model_name in available_models:
                return True
                
            # Check if model name is a prefix of any available model
            for model in available_models:
                if model.startswith(self.model_name) or self.model_name in model:
                    print(f"Found similar model: {model}, using it instead of {self.model_name}")
                    self.model_name = model
                    return True
                    
            print(f"Warning: Model {self.model_name} not found. Available models: {available_models}")
            return False
        except requests.RequestException as e:
            print(f"Warning: Cannot check available models: {str(e)}")
            return False
    
    def generate(self, prompt: str, params: Dict[Any, Any]) -> str:
        """Generate a response from the model"""
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": params.get('creativity', 0.5),
                "stream": False,
                "format": params.get('format', 'text')
            }
            
            # Add optional parameters only if they exist
            if 'context' in params and params['context']:
                payload["context"] = params['context']
            if 'system_prompt' in params and params['system_prompt']:
                payload["system"] = params['system_prompt']
                
            print(f"Sending request to Ollama with model: {self.model_name}")
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=60  # Longer timeout for model inference
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

class OpenAIEngine(InferenceEngine):
    """OpenAI API inference engine implementation"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
    
    @staticmethod
    def get_available_models() -> list:
        """Get list of available OpenAI models"""
        # These are the standard OpenAI models
        return [
            "gpt-3.5-turbo", 
            "gpt-4", 
            "gpt-4-turbo", 
            "gpt-4o",
            "gpt-4-vision"
        ]
    
    def generate(self, prompt: str, params: Dict[Any, Any]) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
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
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except (KeyError, json.JSONDecodeError, requests.RequestException) as e:
            return f"Error with OpenAI inference: {str(e)}"


class ClaudeEngine(InferenceEngine):
    """Claude API inference engine implementation"""
    
    def __init__(self, model_name: str = "claude-2"):
        self.model_name = model_name
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    @staticmethod
    def get_available_models() -> list:
        """Get list of available Claude models"""
        # These are the standard Claude models
        return ["claude-2", "claude-instant-1", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
    
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
