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
            if response.status_code != 200:
                print(f"Warning: Ollama API returned status {response.status_code}")
                return False
                
            data = response.json()
            if 'models' not in data:
                print(f"Warning: Unexpected response format from Ollama API: {data}")
                return False
                
            available_models = [model['name'] for model in data['models']]
            print(f"DEBUG: Available Ollama models: {available_models}")
            
            # Check if the exact model name exists
            if self.model_name in available_models:
                print(f"DEBUG: Found exact model match: {self.model_name}")
                return True
                
            # For models with namespaces (containing slashes) or tags (containing colons)
            # We need to be more flexible in matching
            model_base = self.model_name.split(':')[0]  # Remove tag if present
            model_base = model_base.split('/')[-1]  # Get just the model name without namespace
            
            print(f"DEBUG: Looking for model base: {model_base}")
            
            # Try matching with more flexibility
            for model in available_models:
                # Check if model name is a prefix of any available model
                if model.startswith(self.model_name) or self.model_name in model:
                    print(f"Found similar model: {model}, using it instead of {self.model_name}")
                    self.model_name = model
                    return True
                
                # Check if the base model name matches
                if model_base and (model_base in model or model.endswith(model_base)):
                    print(f"Found model with matching base name: {model}, using it instead of {self.model_name}")
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
            # Verify model is available before attempting to use it
            if not self._check_model():
                return f"Error: Model '{self.model_name}' is not available in Ollama"
            
            # Prepare request payload
            # Ensure creativity/temperature is a float value
            creativity = params.get('creativity', 0.5)
            if not isinstance(creativity, (int, float)):
                try:
                    creativity = float(creativity)
                except (ValueError, TypeError):
                    creativity = 0.5
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": creativity,
                "stream": False
            }
            
            # Debug information
            print(f"DEBUG: Using Ollama model: {self.model_name}")
            
            # For alignment requests, don't add format
            # For main requests, don't force JSON format as it's causing issues
            if params.get('is_alignment', False):
                # Don't add format for alignment requests
                if 'format' in payload:
                    del payload['format']
            # Don't force JSON format for main requests either
                
            print(f"DEBUG: Request payload: {json.dumps(payload, indent=2)}")
            
            # Add optional parameters only if they exist
            if 'context' in params and params['context']:
                payload["context"] = params['context']
            if 'system_prompt' in params and params['system_prompt']:
                payload["system"] = params['system_prompt']
                
            print(f"Sending request to Ollama with model: {self.model_name}")
            try:
                response = requests.post(
                    self.generate_url,
                    json=payload,
                    timeout=180  # Even longer timeout for model inference
                )
                print(f"DEBUG: Ollama response status: {response.status_code}")
            except Exception as e:
                print(f"DEBUG: Ollama request exception: {str(e)}")
                raise
            response.raise_for_status()
            result = response.json()
            if 'response' not in result:
                print(f"DEBUG: Unexpected Ollama response format: {result}")
                return f"Error: Unexpected response format from Ollama"
            # Extract the response based on format and request type
            if 'response' in result:
                response_text = result['response']
                print(f"DEBUG: Extracted response from Ollama: {response_text[:100]}...")
                
                # Try to parse JSON responses regardless of format parameter
                try:
                    # Check if the response looks like JSON
                    if response_text and response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                        print(f"DEBUG: Response appears to be JSON, attempting to parse")
                        json_obj = json.loads(response_text)
                        if isinstance(json_obj, dict):
                            # Extract the content from the JSON
                            if 'response' in json_obj:
                                print(f"DEBUG: Extracted 'response' field from JSON")
                                return json_obj['response']
                            elif 'content' in json_obj:
                                print(f"DEBUG: Extracted 'content' field from JSON")
                                return json_obj['content']
                            elif 'message' in json_obj:
                                print(f"DEBUG: Extracted 'message' field from JSON")
                                return json_obj['message']
                            elif 'text' in json_obj:
                                print(f"DEBUG: Extracted 'text' field from JSON")
                                return json_obj['text']
                            else:
                                print(f"DEBUG: No standard fields found in JSON response")
                                # If it's an empty object or we can't find known fields
                                if not json_obj or json_obj == {}:
                                    print(f"DEBUG: Empty JSON object detected")
                                    return "The model returned an empty response. Please try again."
                                # Return the stringified JSON as fallback
                                return json.dumps(json_obj, indent=2)
                    # If it doesn't look like JSON or parsing fails, return as is
                    return response_text
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON parsing error: {str(e)}")
                    # Not valid JSON, return as is
                    return response_text
                
                return response_text
            else:
                print(f"DEBUG: Unexpected Ollama response format: {result}")
                return f"Error: Unexpected response format from Ollama"
        except requests.HTTPError as e:
            error_msg = f"Model {self.model_name} not found"
            try:
                error_json = e.response.json()
                if 'error' in error_json:
                    error_msg = f"Ollama error: {error_json['error']}"
            except:
                pass
                
            if e.response.status_code == 404:
                return error_msg
            elif e.response.status_code == 500:
                return f"Internal Ollama server error: {error_msg}"
            else:
                return f"HTTP error occurred: {str(e)} - {error_msg}"
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
