from typing import Dict, Any
import requests

class PromptProcessor:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def process(self, prompt: str, params: Dict[Any, Any]) -> str:
        # Your preprocessing logic here
        response = requests.post(
            self.api_url,
            json={
                "model": self.model_name,
                "prompt": prompt
            }
        )
        return response.json()['response']