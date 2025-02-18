from typing import Dict, Any
import requests
import json

class PromptProcessor:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def process(self, prompt: str, params: Dict[Any, Any]) -> str:
        # Extract parameters
        style = params.get('style', 'Professional')
        tone = params.get('tone', 'Neutral')
        creativity = params.get('creativity', 0.5)

        # Construct the system message with parameters
        system_context = (
            f"Respond in a {style.lower()} style with a {tone.lower()} tone. "
            f"Use a creativity level of {creativity}."
        )

        # Combine system context with the prompt
        full_prompt = f"{system_context}\n\nUser: {prompt}"

        # Make API request
        response = requests.post(
            self.api_url,
            json={
                "model": self.model_name,
                "prompt": full_prompt,
                "temperature": creativity  # Map creativity to temperature
            }
        )

        try:
            return response.json()['response']
        except (KeyError, json.JSONDecodeError) as e:
            return f"Error processing request: {str(e)}"