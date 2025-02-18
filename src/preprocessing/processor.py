from typing import Dict, Any, Tuple
from ..inferencing.inference import InferenceEngine

class PromptProcessor:
    def __init__(self):
        """Initialize with two inference engines - one for alignment and one for main processing"""
        self.alignment_engine = InferenceEngine.create_engine("ollama", "huihui_ai/llama3.2-abliterated")
        self.main_engine = None  # Will be set based on user selection

    def set_main_engine(self, engine_type: str, model_name: str):
        """Set the main processing engine based on user selection"""
        self.main_engine = InferenceEngine.create_engine(engine_type, model_name)

    def process_alignment(self, alignment_text: str, params: Dict[Any, Any]) -> str:
        """Process the alignment text with the Llama model"""
        # TODO: Improve alignment steps logic
        system_context = (
            "You are an AI alignment specialist. Analyze the following text "
            "and provide a concise interpretation that will help guide the main model's behavior."
        )
        full_prompt = f"{system_context}\n\nText to analyze: {alignment_text}"
        return self.alignment_engine.generate(full_prompt, params)

    def preprocess_main_prompt(self, prompt: str) -> str:
        """Preprocess the main prompt before sending to the model"""
        # Placeholder for text preprocessing
        # TODO: Implement actual preprocessing logic
        return prompt

    def process_main(self, prompt: str, alignment_result: str, params: Dict[Any, Any]) -> str:
        """Process the main prompt using the alignment result"""
        if not self.main_engine:
            raise ValueError("Main engine not set. Call set_main_engine first.")

        # Preprocess the prompt
        processed_prompt = self.preprocess_main_prompt(prompt)

        style = params.get('style', 'Professional')
        tone = params.get('tone', 'Neutral')
        creativity = params.get('creativity', 0.5)

        system_context = (
            f"Respond in a {style.lower()} style with a {tone.lower()} tone. "
            f"Use a creativity level of {creativity}.\n\n"
            f"Consider this alignment context: {alignment_result}\n\n"
            "The user has reviewed and approved this alignment. "
            "Please provide a response that takes this alignment into account."
        )

        full_prompt = f"{system_context}\n\nUser: {processed_prompt}"
        return self.main_engine.generate(full_prompt, params)
