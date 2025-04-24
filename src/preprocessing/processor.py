from typing import Dict, Any, Tuple, Generator, Union
from src.inferencing.inference import InferenceEngine
import logging
import json

logger = logging.getLogger(__name__)

class PromptProcessor:
    def __init__(self):
        """Initialize with two inference engines - one for alignment and one for main processing"""
        try:
            # Use a standard Ollama model name (llama3 is the correct format)
            self.alignment_engine = InferenceEngine.create_engine("ollama", "llama3")
        except ConnectionError:
            # Fallback to OpenAI if Ollama is not available
            try:
                self.alignment_engine = InferenceEngine.create_engine("openai", "gpt-3.5-turbo") # Or another default
            except Exception as e:
                logger.warning(f"Could not initialize any alignment engine: {str(e)}")
                # No fallback available, will need to be set later
                self.alignment_engine = None # UI should handle this case
        try:
             # Initialize main_engine with a default as well, can be overridden
             self.main_engine = InferenceEngine.create_engine("ollama", "llama3")
        except ConnectionError:
             try:
                 self.main_engine = InferenceEngine.create_engine("openai", "gpt-3.5-turbo")
             except Exception as e:
                 logger.warning(f"Could not initialize any main engine: {str(e)}")
                 self.main_engine = None # UI should handle this case

    def set_main_engine(self, engine_type: str, model_name: str):
        """Set the main processing engine based on user selection"""
        try:
            self.main_engine = InferenceEngine.create_engine(engine_type, model_name)
            logger.info(f"Main engine set to: {engine_type} - {model_name}")
        except ValueError as e:
             logger.error(f"Failed to set main engine ({engine_type}, {model_name}): {e}")
             # Keep the previous engine or set to None? For now, keep previous.
             # self.main_engine = None
             raise # Re-raise the error to be caught by the UI
        except Exception as e:
             logger.error(f"Unexpected error setting main engine: {e}")
             raise

    def process_alignment(self, alignment_text: str, params: Dict[Any, Any]) -> Generator[str, None, None]:
        """Process the alignment text with the alignment engine, yielding results."""
        if not self.alignment_engine:
            def error_gen(): yield "Error: No alignment engine available"
            return error_gen()

        # Handle image prompt separately
        if "image_base64" in params and params["image_base64"]:
            if not alignment_text: # Use default prompt if no text provided with image
                 alignment_text = "Describe the style, tone, and values implied by this image."
            system_context = (
                "You are an AI alignment specialist. Analyze the uploaded image and the accompanying text (if any). "
                "Describe what values, tone, or style it implies. "
                "Return guidance that can help a language model match that intention. Respond with plain text."
            )
            full_prompt = f"{system_context}\n\nText (optional context): {alignment_text}"
            logger.info("Processing image alignment...")
        elif not alignment_text or not alignment_text.strip():
             def error_gen(): yield "Please provide alignment text or an image to analyze"
             return error_gen()
        else:
            # Standard text alignment
            system_context = (
                "You are an AI alignment specialist. Analyze the following text "
                "and provide a concise interpretation that will help guide the main model's behavior. "
                "Focus on extracting key principles, constraints, and goals from the text. "
                "Respond with plain text, not JSON."
            )
            full_prompt = f"{system_context}\n\nText to analyze: {alignment_text}"
            logger.info("Processing text alignment...")

        # Prepare parameters for the alignment engine
        alignment_params = params.copy()
        # Ensure 'format' is not included unless specifically needed for alignment
        alignment_params.pop('format', None)
        # Mark as alignment request (might influence engine behavior)
        alignment_params['is_alignment'] = True
        # Add system prompt if not already part of the full_prompt logic
        # alignment_params['system_prompt'] = system_context # If engine uses this param

        # Generate the response using streaming
        response_generator = self.alignment_engine.generate(
            prompt=full_prompt,
            params=alignment_params,
            stream=True
        )

        # Yield chunks from the generator
        try:
            for chunk in response_generator:
                yield chunk
        except Exception as e:
            logger.error(f"Error consuming alignment stream: {e}")
            yield f"\nError during alignment streaming: {e}\n"

    # --- optimize_prompt remains non-streaming ---
    # It needs the full alignment result to work with.
    # It calls generate(stream=False) internally.

    def preprocess_main_prompt(self, prompt: str) -> str:
        """Preprocess the main prompt before sending to the model"""
        # Basic preprocessing steps:
        # 1. Trim whitespace
        processed = prompt.strip()

        # 2. Ensure the prompt ends with a question mark if it seems like a question
        question_starters = ["what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "should", "is", "are", "do", "does"]
        words = processed.lower().split()
        if words and words[0] in question_starters and not processed.endswith("?"):
            processed += "?"

        # 3. Add a polite prefix if the prompt is very short (likely a command)
        if len(processed.split()) < 4 and not any(q in processed.lower() for q in question_starters):
            processed = f"Please {processed.lower()}"

        # 4. Ensure proper capitalization
        if processed and processed[0].islower():
            processed = processed[0].upper() + processed[1:]

        return processed

    def optimize_prompt(self, original_prompt: str, alignment_result: str, params: Dict[Any, Any]) -> str:
        """Use the alignment engine to optimize the user's prompt based on alignment principles"""
        """Use the alignment engine to optimize the user's prompt based on alignment principles"""
        if not self.alignment_engine or not alignment_result or not alignment_result.strip():
            logger.debug("Skipping prompt optimization (no engine or alignment result).")
            return original_prompt

        # Create a copy of params for the alignment engine call
        optimization_params = params.copy()
        optimization_params['is_alignment'] = True # Mark as internal task
        optimization_params.pop('format', None)
        optimization_params.pop('image_base64', None) # Image not relevant here

        system_context = (
            "You are an AI prompt optimizer. Your job is to SUBTLY enhance the user's prompt "
            "while maintaining its original intent and purpose.\n\n"
            f"Alignment context: {alignment_result}\n\n"
            "Rules for optimization:\n"
            "1. The user's original intent and question MUST be preserved as the primary focus\n"
            "2. Make only minimal, necessary additions to the prompt\n"
            "3. You may add BRIEF context from alignment principles only if directly relevant\n"
            "4. Do not change the core request or question\n"
            "5. DO NOT answer the prompt yourself, just optimize it\n"
            "6. Return ONLY the optimized prompt text, nothing else (no preamble, no explanation).\n"
            "7. If the original prompt already aligns well or optimization is not applicable, return the original prompt text unchanged.\n"
            "8. Keep the optimized prompt concise and focused on the user's core request.\n"
        )

        full_prompt = f"{system_context}\n\nOriginal prompt: ```\n{original_prompt}\n```\n\nOptimized prompt:"
        logger.debug("Requesting prompt optimization from alignment engine.")

        # Get the optimized prompt from the alignment engine (non-streaming)
        try:
            optimized_prompt = self.alignment_engine.generate(
                prompt=full_prompt,
                params=optimization_params,
                stream=False # We need the full result here
            )
        except Exception as e:
             logger.error(f"Error during prompt optimization generation: {e}")
             return original_prompt # Fallback to original on error

        # Clean up the response - remove potential markdown, quotes, preamble
        optimized_prompt = str(optimized_prompt).strip() # Ensure string type
        if optimized_prompt.startswith("Optimized prompt:"):
             optimized_prompt = optimized_prompt[len("Optimized prompt:"):].strip()
        if optimized_prompt.startswith("```") and optimized_prompt.endswith("```"):
             optimized_prompt = optimized_prompt[3:-3].strip()
        if optimized_prompt.startswith('"') and optimized_prompt.endswith('"'):
             optimized_prompt = optimized_prompt[1:-1].strip()


        # Basic validation: If optimization failed, returned empty, or is identical, use the original
        if not optimized_prompt or len(optimized_prompt) < 5 or optimized_prompt == original_prompt:
            logger.debug("Prompt optimization resulted in empty, too short, or identical prompt. Using original.")
            return original_prompt

        # Heuristic: If the optimized prompt is drastically different (e.g., much shorter/longer), maybe stick to original.
        # len_orig = len(original_prompt)
        # len_opt = len(optimized_prompt)
        # if len_opt < len_orig * 0.5 or len_opt > len_orig * 2.0:
        #     logger.warning("Optimized prompt length significantly different, potentially problematic. Using original.")
        #     return original_prompt

        logger.debug(f"Original prompt: {original_prompt}")
        logger.info(f"Using optimized prompt: {optimized_prompt}")

        return optimized_prompt


    def process_main(self, prompt: str, alignment_result: str, params: Dict[Any, Any]) -> Generator[str, None, None]:
        """Process the main prompt using the alignment result, yielding results."""
        if not self.main_engine:
            def error_gen(): yield "Error: Main processing engine is not configured."
            logger.error("process_main called but main_engine is not set.")
            return error_gen()

        # Ensure this is marked as a main request (not alignment)
        main_params = params.copy()
        main_params['is_alignment'] = False # Ensure it's marked as a main request
        main_params.pop('image_base64', None) # Image handled by alignment, not usually sent to main prompt directly unless model supports it

        # Preprocess the user's prompt text (basic cleaning)
        processed_prompt = self.preprocess_main_prompt(prompt)

        # Optimize the preprocessed prompt using the alignment result (non-streaming)
        # Pass original params to optimize_prompt as it might use creativity etc.
        optimized_prompt = self.optimize_prompt(processed_prompt, alignment_result, params)

        # Extract values from params
        style = str(params.get('style', 'Professional'))
        tone = str(params.get('tone', 'Neutral'))

        # Ensure creativity is a float
        try:
            creativity = float(params.get('creativity', 0.5))
        except (ValueError, TypeError):
            creativity = 0.5
        main_params['creativity'] = creativity # Ensure it's in the params passed to generate

        logger.debug(f"Main processing using style={style}, tone={tone}, creativity={creativity}")

        # Construct the system prompt, incorporating alignment if available
        system_parts = []
        # Style/Tone instructions (optional, can sometimes conflict with alignment)
        # system_parts.append(f"Respond in a {style.lower()} style with a {tone.lower()} tone.")

        if alignment_result and alignment_result.strip():
             system_parts.append(
                 "--- Alignment Guidance (Consider this as secondary context) ---\n"
                 f"{alignment_result}\n"
                 "--- End Alignment Guidance ---"
             )
        system_parts.append(
            "Focus on fulfilling the user's request below. The alignment guidance should influence *how* you respond (e.g., tone, style, constraints) but not *what* you respond about unless directly relevant. Prioritize the user's specific request."
        )
        system_prompt = "\n\n".join(system_parts)

        # Add system prompt to params if engine supports it (Ollama 'system', OpenAI 'system' message)
        main_params['system_prompt'] = system_prompt

        # Use the optimized prompt as the user's message
        final_user_prompt = optimized_prompt

        logger.info("Generating main response (streaming)...")
        # Generate the response using streaming
        response_generator = self.main_engine.generate(
            prompt=final_user_prompt,
            params=main_params,
            stream=True
        )

        # Yield chunks from the generator
        try:
            for chunk in response_generator:
                yield chunk
        except Exception as e:
            logger.error(f"Error consuming main response stream: {e}")
            yield f"\nError during main response streaming: {e}\n"
