from typing import Dict, Any
from ..inferencing.inference import InferenceEngine, OllamaEngine, ClaudeEngine

class PromptProcessor:
    def __init__(self, engine: str = "ollama", model_name: str = "mistral"):
        """Initialize with specified engine type and model name"""
        if engine == "ollama":
            self.engine = OllamaEngine(model_name)
        elif engine == "claude":
            self.engine = ClaudeEngine(model_name)
        else:
            raise ValueError(f"Unsupported engine type: {engine}")

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

        # Use the inference engine to generate response
        return self.engine.generate(full_prompt, params)
