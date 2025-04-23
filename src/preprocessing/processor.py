from typing import Dict, Any, Tuple
from src.inferencing.inference import InferenceEngine
import requests
from bs4 import BeautifulSoup


def html_url_to_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Grab visible paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]

        # Grab table content (especially from Wikipedia)
        tables = []
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if cells:
                    row_text = " | ".join(cell.get_text(strip=True) for cell in cells)
                    tables.append(row_text)

        combined = paragraphs + tables
        return "\n".join(combined).strip() or "No readable content found on the page."
    except Exception as e:
        raise RuntimeError(f"Failed to fetch or parse URL: {url} — {str(e)}")


def preprocess_main_prompt(prompt: str) -> str:
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


class PromptProcessor:
    def __init__(self):
        """Initialize with two inference engines - one for alignment and one for main processing"""
        try:
            # Use a standard Ollama model name (llama3 is the correct format)
            self.alignment_engine = InferenceEngine.create_engine("ollama", "llama3")
        except ConnectionError:
            # Fallback to OpenAI if Ollama is not available
            try:
                self.alignment_engine = InferenceEngine.create_engine("openai", "gpt-3.5-turbo")
            except Exception as e:
                print(f"Warning: Could not initialize any alignment engine: {str(e)}")
                # No fallback available, will need to be set later
                self.alignment_engine = None
        self.main_engine = None  # Will be set based on user selection

    def set_main_engine(self, engine_type: str, model_name: str):
        """Set the main processing engine based on user selection"""
        self.main_engine = InferenceEngine.create_engine(engine_type, model_name)

    def html_url_to_text(self, url: str) -> str:
        """Fetch HTML content and extract visible text, including from tables."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            tables = []
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if cells:
                        row_text = " | ".join(cell.get_text(strip=True) for cell in cells)
                        tables.append(row_text)
            combined = paragraphs + tables
            return "\n".join(combined).strip() or "No readable content found on the page."
        except Exception as e:
            raise RuntimeError(f"Failed to fetch or parse URL: {url} — {str(e)}")

    def process_alignment(self, alignment_text: str, params: Dict[Any, Any]) -> str:
        """Process the alignment text with the Llama model"""
        if not self.alignment_engine:
            return "Error: No alignment engine available"

        if not alignment_text or alignment_text.strip() == "":
            return "Please provide alignment text to analyze"

        # TODO: Improve alignment steps logic
        system_context = (
            "You are an AI alignment specialist. Analyze the following text "
            "and provide a concise interpretation that will help guide the main model's behavior. "
            "Focus on extracting key principles, constraints, and goals from the text. "
            "Respond with plain text, not JSON."
        )
        full_prompt = f"{system_context}\n\nText to analyze: {alignment_text}"

        # Create a copy of params and remove any format parameter
        alignment_params = params.copy()
        if 'format' in alignment_params:
            del alignment_params['format']  # Remove format parameter entirely

        # Mark this as an alignment request and get the response
        alignment_params['is_alignment'] = True
        response = self.alignment_engine.generate(full_prompt, alignment_params)

        if "image_base64" in params:
            full_prompt = (
                "You are an AI alignment specialist. Analyze the uploaded image and describe what values, tone, or style it implies. "
                "Then return guidance that can help a language model match that intention."
            )
        else:
            full_prompt = (
                "You are an AI alignment specialist. Analyze the following text "
                "and provide a concise interpretation that will help guide the main model's behavior. "
                "Focus on extracting key principles, constraints, and goals from the text. "
                "Respond with plain text, not JSON.\n\n"
                f"Text to analyze: {alignment_text}"
            )

        # Handle potential JSON responses
        import json
        try:
            # Check if the response is JSON
            if response and response.strip().startswith('{') and response.strip().endswith('}'):
                print(f"DEBUG: Detected JSON response: {response[:100]}...")
                json_response = json.loads(response)
                # Extract content from common JSON response formats
                if isinstance(json_response, dict):
                    if 'response' in json_response:
                        response = json_response['response']
                        print(f"DEBUG: Extracted 'response' field: {response[:100]}...")
                    elif 'content' in json_response:
                        response = json_response['content']
                        print(f"DEBUG: Extracted 'content' field: {response[:100]}...")
                    elif 'message' in json_response:
                        response = json_response['message']
                        print(f"DEBUG: Extracted 'message' field: {response[:100]}...")
                    elif 'text' in json_response:
                        response = json_response['text']
                        print(f"DEBUG: Extracted 'text' field: {response[:100]}...")
                    else:
                        # If we can't find a known field, convert the whole JSON to a string
                        print(f"DEBUG: No known fields found in JSON, using stringified version")
                        response = json.dumps(json_response, indent=2)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # Not valid JSON or other error, use the response as is
            print(f"DEBUG: JSON parsing error: {str(e)}")

        # If the response is empty or just contains {}, return a helpful message
        if not response or response.strip() in ["", "{}", "{", "}"]:
            print(f"DEBUG: Empty or invalid response detected: '{response}'")
            return "The alignment engine did not provide a valid response. Please try again."

        print(f"DEBUG: Final alignment response: {response[:100]}...")

        return response

    def optimize_prompt(self, original_prompt: str, alignment_result: str, params: Dict[Any, Any]) -> str:
        """Use the alignment engine to optimize the user's prompt based on alignment principles"""
        if not self.alignment_engine:
            return original_prompt

            # Create a copy of params for the alignment engine
        alignment_params = params.copy()
        alignment_params['is_alignment'] = True

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
            "6. Return ONLY the optimized prompt text, nothing else\n"
            "7. If the original prompt already aligns well, return it unchanged\n"
        )

        full_prompt = f"{system_context}\n\nOriginal prompt: {original_prompt}\n\nOptimized prompt:"

        # Get the optimized prompt from the alignment engine
        optimized_prompt = self.alignment_engine.generate(full_prompt, alignment_params)


        # Clean up the response
        optimized_prompt = optimized_prompt.strip()

        # If optimization failed or returned empty, use the original
        if not optimized_prompt or len(optimized_prompt) < 5:
            print("DEBUG: Prompt optimization failed, using original prompt")
            return original_prompt

            # If the optimized prompt is too different from the original, use the original
        # This is a simple heuristic to prevent excessive changes
        if len(optimized_prompt) > len(original_prompt) * 2:
            print("DEBUG: Optimized prompt too different from original, using original prompt")
            return original_prompt

        print(f"DEBUG: Original prompt: {original_prompt}")
        print(f"DEBUG: Optimized prompt: {optimized_prompt}")

        return optimized_prompt

    def process_main(self, prompt: str, alignment_result: str, params: Dict[Any, Any]) -> str:
        """Process the main prompt using the alignment result"""
        if not self.main_engine:
            raise ValueError("Main engine not set. Call set_main_engine first.")

        # Ensure this is marked as a main request (not alignment)
        main_params = params.copy()
        main_params['is_alignment'] = False

        # Preprocess the prompt
        processed_prompt = preprocess_main_prompt(prompt)

        # Optimize the prompt using the alignment engine
        optimized_prompt = self.optimize_prompt(processed_prompt, alignment_result, params)

        # Inject webpage content if URL was provided
        webpage_text = ""
        if "url" in params and params["url"]:
            try:
                webpage_text = html_url_to_text(params["url"])
                print("DEBUG: Extracted webpage text successfully.")
            except Exception as e:
                print(f"DEBUG: Failed to extract webpage text: {e}")

        # Extract values from params
        style = str(params.get('style', 'Professional'))
        tone = str(params.get('tone', 'Neutral'))

        # Ensure creativity is a float
        try:
            creativity = float(params.get('creativity', 0.5))
        except (ValueError, TypeError):
            creativity = 0.5

        print(f"DEBUG: Using style={style}, tone={tone}, creativity={creativity}")

        system_context = (
            f"Consider this alignment context as secondary guidance: {alignment_result}\n\n"
            "The user's request is your primary objective. The alignment context should "
            "only influence HOW you respond, not WHAT you respond about. "
            "Always prioritize addressing the user's specific request."
        )

        # Inject the webpage content if available
        if webpage_text:
            system_context += f"\n\n[BEGIN WEBPAGE CONTENT]\n{webpage_text[:3000]}\n[END WEBPAGE CONTENT]"

        full_prompt = f"{system_context}\n\n{optimized_prompt}"

        # Generate the response
        response = self.main_engine.generate(full_prompt, main_params)

        # Handle potential JSON parsing issues
        import json
        try:
            # Check if the response is JSON
            if response and response.strip().startswith('{') and response.strip().endswith('}'):
                print(f"DEBUG: Main model returned JSON response: {response[:100]}...")
                json_response = json.loads(response)
                # Extract content from common JSON response formats
                if isinstance(json_response, dict):
                    if 'response' in json_response:
                        response = json_response['response']
                        print(f"DEBUG: Extracted 'response' field from main model: {response[:100]}...")
                    elif 'content' in json_response:
                        response = json_response['content']
                        print(f"DEBUG: Extracted 'content' field from main model: {response[:100]}...")
                    elif 'message' in json_response:
                        response = json_response['message']
                        print(f"DEBUG: Extracted 'message' field from main model: {response[:100]}...")
                    elif 'text' in json_response:
                        response = json_response['text']
                        print(f"DEBUG: Extracted 'text' field from main model: {response[:100]}...")
                    else:
                        # If we can't find a known field, convert the whole JSON to a string
                        print(f"DEBUG: No known fields found in main model JSON, using stringified version")
                        response = json.dumps(json_response, indent=2)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # Not valid JSON or other error, use the response as is
            print(f"DEBUG: Main model JSON parsing error: {str(e)}")

        # If the response is empty or just contains {}, return a helpful message
        if not response or response.strip() in ["", "{}", "{", "}", "[]"]:
            print(f"DEBUG: Empty or invalid main model response detected: '{response}'")
            return "The model did not provide a valid response. Please try again with different parameters or prompt."

        print(f"DEBUG: Final main model response: {response[:100]}...")
        return response
