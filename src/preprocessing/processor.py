from typing import Dict, Any, Tuple, Generator, Union
from src.inferencing.inference import InferenceEngine, OllamaEngine
import logging
import json
import requests
from urllib.parse import urlparse

# Try to import trafilatura, but provide fallback if not available
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logging.getLogger(__name__).warning("trafilatura module not found. URL extraction will use basic fallback method.")

logger = logging.getLogger(__name__)

class PromptProcessor:
    def __init__(self):
        """Initialize with two inference engines - one for alignment and one for main processing"""
        try:
            # Try to get available Ollama models first
            available_models = []
            try:
                available_models = OllamaEngine.get_available_models()
            except Exception as e:
                logger.warning(f"Could not get available Ollama models: {str(e)}")
            
            # Use llama3:latest if available, otherwise try the first available model
            if "llama3:latest" in available_models:
                self.alignment_engine = InferenceEngine.create_engine("ollama", "llama3:latest")
            elif available_models:
                logger.info(f"Using first available Ollama model: {available_models[0]}")
                self.alignment_engine = InferenceEngine.create_engine("ollama", available_models[0])
            else:
                # Default to llama3 if we couldn't get the model list
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
             # Use the same logic as for alignment engine
             available_models = []
             try:
                 available_models = OllamaEngine.get_available_models()
             except Exception as e:
                 logger.warning(f"Could not get available Ollama models: {str(e)}")
             
             if "llama3:latest" in available_models:
                 self.main_engine = InferenceEngine.create_engine("ollama", "llama3:latest")
             elif available_models:
                 logger.info(f"Using first available Ollama model: {available_models[0]}")
                 self.main_engine = InferenceEngine.create_engine("ollama", available_models[0])
             else:
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
            
        # Check for URL in params and extract content if present
        url = params.get("alignment_url", "")
        if url and url.strip():
            logger.info(f"Extracting content from URL: {url}")
            extracted_content = self.extract_content_from_url(url)
            
            # If extraction returned an error message
            if extracted_content.startswith("Error:"):
                def error_gen(): yield extracted_content
                return error_gen()
                
            # If we have both URL content and user-provided text, combine them
            if alignment_text and alignment_text.strip():
                alignment_text = f"{alignment_text}\n\n--- Content extracted from {url} ---\n\n{extracted_content}"
            else:
                alignment_text = extracted_content
                
            logger.info(f"Using extracted content from URL ({len(extracted_content)} chars)")

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
             def error_gen(): yield "Please provide alignment text, an image, or a URL to analyze"
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

    def extract_content_from_url(self, url: str) -> str:
        """Extract main article content from a URL using Trafilatura or fallback method"""
        if not url or not url.strip():
            return ""
            
        # Basic URL validation
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.warning(f"Invalid URL format: {url}")
                return f"Error: Invalid URL format. Please provide a complete URL including http:// or https://"
        except Exception as e:
            logger.error(f"URL parsing error: {e}")
            return f"Error: Could not parse URL: {str(e)}"
        
        # Check if trafilatura is available
        if not TRAFILATURA_AVAILABLE:
            return self._extract_content_fallback(url)
            
        try:
            logger.info(f"Fetching content from URL using Trafilatura: {url}")
            # Download the webpage content
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                logger.warning(f"Failed to download content from URL: {url}")
                return f"Error: Could not download content from {url}"
                
            # Extract the main content
            extracted_text = trafilatura.extract(downloaded, include_links=False, 
                                               include_images=False, 
                                               include_tables=False,
                                               output_format='text')
            
            if not extracted_text or not extracted_text.strip():
                logger.warning(f"No content extracted from URL: {url}")
                # Try fallback method
                return self._extract_content_fallback(url)
                
            # Trim and clean up the extracted text
            cleaned_text = extracted_text.strip()
            
            # Limit length if extremely long
            max_length = 8000  # Reasonable limit to avoid token issues
            if len(cleaned_text) > max_length:
                logger.info(f"Truncating extracted content from {len(cleaned_text)} to {max_length} chars")
                cleaned_text = cleaned_text[:max_length] + "...\n[Content truncated due to length]"
                
            logger.info(f"Successfully extracted {len(cleaned_text)} chars from URL")
            return cleaned_text
            
        except Exception as e:
            logger.exception(f"Error extracting content from URL with Trafilatura: {e}")
            # Try fallback method
            return self._extract_content_fallback(url)
            
    def _extract_content_fallback(self, url: str) -> str:
        """Basic fallback method for extracting content when Trafilatura is not available"""
        try:
            logger.info(f"Using fallback extraction method for URL: {url}")
            
            # Simple request to get the page content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Get the HTML content
            html_content = response.text
            
            # Very basic extraction - get text between <p> tags
            import re
            paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
            
            # Clean up the paragraphs (remove HTML tags)
            cleaned_paragraphs = []
            for p in paragraphs:
                # Remove HTML tags
                clean_p = re.sub(r'<[^>]+>', '', p)
                # Remove extra whitespace
                clean_p = re.sub(r'\s+', ' ', clean_p).strip()
                if clean_p and len(clean_p) > 20:  # Only keep paragraphs with substantial content
                    cleaned_paragraphs.append(clean_p)
            
            # Join the paragraphs
            extracted_text = '\n\n'.join(cleaned_paragraphs)
            
            if not extracted_text or not extracted_text.strip():
                logger.warning(f"No content extracted from URL with fallback method: {url}")
                return f"Error: Could not extract content from {url}. Please install trafilatura for better extraction."
            
            # Limit length if extremely long
            max_length = 8000
            if len(extracted_text) > max_length:
                logger.info(f"Truncating extracted content from {len(extracted_text)} to {max_length} chars")
                extracted_text = extracted_text[:max_length] + "...\n[Content truncated due to length]"
            
            logger.info(f"Successfully extracted {len(extracted_text)} chars from URL using fallback method")
            
            # Add a note about using fallback method
            return extracted_text + "\n\n[Note: Content extracted using basic method. Install trafilatura for better results.]"
            
        except Exception as e:
            logger.exception(f"Error in fallback extraction method: {e}")
            return f"Error: Failed to extract content: {str(e)}. Please install trafilatura for better extraction."

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
        # Check if alignment result contains an error message
        if not self.alignment_engine or not alignment_result or not alignment_result.strip() or alignment_result.startswith("Error:"):
            if alignment_result and alignment_result.startswith("Error:"):
                logger.warning(f"Skipping prompt optimization due to alignment error: {alignment_result}")
            else:
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

        # Check if alignment result contains an error
        alignment_has_error = alignment_result and alignment_result.strip() and alignment_result.startswith("Error:")
        if alignment_has_error:
            logger.warning(f"Alignment result contains an error: {alignment_result}")
            # Don't use the error message as alignment guidance
            effective_alignment_result = ""
        else:
            effective_alignment_result = alignment_result

        # Optimize the preprocessed prompt using the alignment result (non-streaming)
        # Pass original params to optimize_prompt as it might use creativity etc.
        optimized_prompt = self.optimize_prompt(processed_prompt, effective_alignment_result, params)

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
        
        # Always include style/tone instructions
        system_parts.append(f"Respond in a {style.lower()} style with a {tone.lower()} tone.")
        
        if effective_alignment_result and effective_alignment_result.strip():
             # Make alignment guidance more prominent and directive
             system_parts.append(
                 "YOU MUST FOLLOW THIS ALIGNMENT GUIDANCE:\n"
                 f"{effective_alignment_result}\n"
                 "The above alignment guidance MUST influence your response content, style, and approach."
             )
             
             # Add explicit instruction to not ignore the alignment
             system_parts.append(
                "IMPORTANT: Do NOT respond as if you're a new instance with no context. You HAVE received alignment guidance above."
             )
        elif alignment_has_error:
             # Add a note about alignment failure but don't include the error message in guidance
             system_parts.append(
                "Note: Alignment processing encountered an error, but you should still respond to the user's request normally."
             )
             
        system_parts.append(
            "Focus on fulfilling the user's request while adhering to the alignment guidance if provided. The alignment guidance should strongly influence both *how* you respond and *what* you include in your response when relevant."
        )
        system_prompt = "\n\n".join(system_parts)

        # Add system prompt to params if engine supports it (Ollama 'system', OpenAI 'system' message)
        main_params['system_prompt'] = system_prompt

        # Modify the user prompt to reinforce alignment if needed
        if effective_alignment_result and effective_alignment_result.strip():
            # For some models, including a reminder in the user prompt helps ensure alignment is followed
            final_user_prompt = f"[Remember to follow the alignment guidance provided in the system prompt]\n\n{optimized_prompt}"
        else:
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
