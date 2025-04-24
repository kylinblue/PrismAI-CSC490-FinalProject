from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Generator
import requests
import json
import os
import logging

logger = logging.getLogger(__name__)

class InferenceEngine(ABC):
    """Abstract base class for inference engines"""

    @abstractmethod
    def generate(self, prompt: str, params: Dict[Any, Any], stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt.
            params: Dictionary of generation parameters.
            stream: If True, yield response chunks; otherwise, return the full response.

        Returns:
            Either the full response string or a generator yielding response chunks.
        """
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
            logger.warning(f"Ollama initialization issue: {str(e)}")
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
            logger.warning(f"Cannot fetch Ollama models: {str(e)}")
            return []

    def _check_connection(self):
        """Test connection to Ollama server"""
        try:
            # Use HEAD request for efficiency
            requests.head(self.base_url, timeout=2)
        except requests.RequestException as e:
            logger.warning(f"Cannot connect to Ollama server: {str(e)}")
            raise ConnectionError(f"Cannot connect to Ollama server: {str(e)}")

    def _check_model(self):
        """Verify if the specified model is available"""
        try:
            # Use HEAD request first for a quick check if the server is responsive
            try:
                head_response = requests.head(f"{self.base_url}/tags", timeout=2)
                head_response.raise_for_status()
            except requests.RequestException as head_e:
                logger.warning(f"Ollama /api/tags endpoint not responsive: {head_e}")
                return False # Cannot verify model if endpoint is down

            # If HEAD is okay, proceed with GET
            response = requests.get(f"{self.base_url}/tags", timeout=5)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            if 'models' not in data or not isinstance(data['models'], list):
                logger.warning(f"Unexpected response format from Ollama API /api/tags: {data}")
                return False

            available_models = [model.get('name') for model in data['models'] if model.get('name')]
            logger.debug(f"Available Ollama models: {available_models}")

            # Check if the exact model name (including tag) exists
            if self.model_name in available_models:
                logger.debug(f"Found exact model match: {self.model_name}")
                return True

            # For models with namespaces (containing slashes) or tags (containing colons)
            # We need to be more flexible in matching
            # Check if model name without tag exists (e.g., user specified 'llama3', server has 'llama3:latest')
            model_name_no_tag = self.model_name.split(':')[0]
            if model_name_no_tag != self.model_name: # Only check if there was a tag originally
                for model in available_models:
                    if model.split(':')[0] == model_name_no_tag:
                        logger.info(f"Found model matching base name: {model}. Using it instead of {self.model_name}")
                        self.model_name = model # Update to the full name found on the server
                        return True

            # More flexible matching (less reliable, kept as last resort)
            # model_base = self.model_name.split(':')[0]
            # model_base = model_base.split('/')[-1]
            # logger.debug(f"Looking for model base: {model_base}")

            # for model in available_models:
            #     # Check if model name is a prefix of any available model
            #     if model.startswith(self.model_name) or self.model_name in model:
            #         logger.info(f"Found similar model: {model}, using it instead of {self.model_name}")
            #         self.model_name = model
            #         return True
            #     # Check if the base model name matches
            #     if model_base and (model_base in model or model.endswith(model_base)):
            #         logger.info(f"Found model with matching base name: {model}, using it instead of {self.model_name}")
            #         self.model_name = model
            #         return True

            logger.warning(f"Model {self.model_name} not found or could not be matched. Available models: {available_models}")
            return False
        except requests.HTTPError as e:
             logger.warning(f"HTTP error checking Ollama models ({e.response.status_code}): {e}")
             return False # Cannot verify model if API gives error
        except requests.RequestException as e:
            logger.warning(f"Network error checking available Ollama models: {str(e)}")
            return False # Cannot verify model if network error

    def generate(self, prompt: str, params: Dict[Any, Any], stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate a response from the Ollama model"""
        # Verify model is available before attempting to use it
        # Use a cached check or less frequent check if performance is an issue
        if not self._check_model():
             # If streaming, yield an error message; otherwise, return it
             error_msg = f"Error: Model '{self.model_name}' is not available in Ollama or connection failed."
             if stream:
                 def error_generator():
                     yield error_msg
                 return error_generator()
             else:
                 return error_msg
        try:

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
                "stream": stream # Use the stream parameter
            }

            # Debug information
            logger.debug(f"Using Ollama model: {self.model_name}")

            # For alignment requests, don't add format
            # For main requests, don't force JSON format as it's causing issues
            if params.get('is_alignment', False):
                # Don't add format for alignment requests
                if 'format' in payload:
                    del payload['format']
            # Don't force JSON format for main requests either (unless specifically requested in params, which isn't currently)

            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")

            # Add optional parameters only if they exist and have values
            if params.get('context'):
                payload["context"] = params['context']
            if params.get('system_prompt'):
                payload["system"] = params['system_prompt']

            logger.info(f"Sending request to Ollama (model: {self.model_name}, stream: {stream})")
            response = requests.post(
                self.generate_url,
                json=payload,
                stream=stream, # Pass stream argument to requests
                timeout=180
            )
            logger.debug(f"Ollama response status: {response.status_code}")
            response.raise_for_status() # Raise HTTPError for bad responses

            if stream:
                return self._process_ollama_stream(response)
            else:
                result = response.json()
                logger.debug(f"Ollama non-stream response received: {str(result)[:200]}...")
                if 'response' not in result:
                    logger.error(f"Unexpected Ollama response format (non-stream): {result}")
                    return f"Error: Unexpected response format from Ollama"
                # Handle potential JSON within the response field (as before)
                return self._parse_ollama_response_content(result['response'])

        except requests.HTTPError as e:
            error_msg = f"Ollama HTTP error ({e.response.status_code})"
            try:
                error_detail = e.response.json().get('error', str(e))
                error_msg += f": {error_detail}"
            except json.JSONDecodeError:
                error_msg += f": {e.response.text}" # Include raw text if not JSON
            logger.error(error_msg)
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg
        except requests.RequestException as e:
            error_msg = f"Network error connecting to Ollama: {str(e)}"
            logger.error(error_msg)
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred during Ollama generation: {str(e)}"
            logger.exception(error_msg) # Log full traceback for unexpected errors
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg

    def _process_ollama_stream(self, response: requests.Response) -> Generator[str, None, None]:
        """Processes the streaming response from Ollama."""
        buffer = ""
        try:
            # Iterate over lines, handling potential bytes chunks
            for chunk in response.iter_lines(): # Get raw bytes first
                if chunk:
                    # Decode the chunk explicitly
                    try:
                        decoded_chunk = chunk.decode('utf-8')
                    except UnicodeDecodeError:
                        logger.warning(f"Ollama stream contained non-UTF-8 data, attempting lossy decode: {chunk}")
                        decoded_chunk = chunk.decode('utf-8', errors='ignore')

                    buffer += decoded_chunk
                    # Ollama streams JSON objects separated by newlines
                    try:
                        # Process lines that form complete JSON objects
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if line.strip():
                                stream_data = json.loads(line)
                                if stream_data.get('done') is False and 'response' in stream_data:
                                    yield stream_data['response']
                                elif stream_data.get('done') is True:
                                    # Optionally process final data like context
                                    # final_context = stream_data.get('context')
                                    logger.debug("Ollama stream finished.")
                                    return # End generation
                        # If buffer has partial JSON, wait for more chunks
                        # Check if remaining buffer might be a complete JSON (less common)
                        if buffer.strip() and buffer.startswith('{') and buffer.endswith('}'):
                             stream_data = json.loads(buffer)
                             if stream_data.get('done') is False and 'response' in stream_data:
                                 yield stream_data['response']
                             buffer = "" # Clear buffer after processing

                    except json.JSONDecodeError:
                        # Incomplete JSON line, wait for the next chunk
                        # Add the newline back if it was prematurely split
                        if not buffer.endswith('\n'):
                             buffer += '\n'
                        logger.debug(f"Incomplete JSON chunk, buffering: {buffer}")
                        continue
                    except Exception as e:
                         logger.error(f"Error processing Ollama stream chunk: {e} - Chunk: {chunk}")
                         yield f"\nError processing stream: {e}\n"

            # Process any remaining buffer after the loop finishes
            if buffer.strip():
                 try:
                     stream_data = json.loads(buffer)
                     if stream_data.get('done') is False and 'response' in stream_data:
                         yield stream_data['response']
                 except json.JSONDecodeError:
                     logger.warning(f"Could not parse final buffer content: {buffer}")
                 except Exception as e:
                     logger.error(f"Error processing final Ollama stream buffer: {e}")
                     yield f"\nError processing final stream part: {e}\n"

        except requests.exceptions.ChunkedEncodingError as e:
            logger.error(f"Ollama stream connection error: {e}")
            yield f"\nStream connection error: {str(e)}\n"
        except Exception as e:
            logger.exception(f"Unexpected error during Ollama stream processing")
            yield f"\nUnexpected stream error: {str(e)}\n"


    def _parse_ollama_response_content(self, response_text: str) -> str:
        """Attempts to parse JSON if response looks like it, otherwise returns text."""
        if not response_text:
            return ""
        stripped_response = response_text.strip()
        if stripped_response.startswith('{') and stripped_response.endswith('}'):
            logger.debug("Response appears to be JSON, attempting to parse")
            try:
                json_obj = json.loads(stripped_response)
                if isinstance(json_obj, dict):
                    # Extract content from common fields
                    for key in ['response', 'content', 'message', 'text']:
                        if key in json_obj:
                            logger.debug(f"Extracted '{key}' field from JSON")
                            return str(json_obj[key]) # Ensure it's a string
                    # Fallback for unknown structure or empty dict
                    if not json_obj:
                        logger.debug("Empty JSON object detected")
                        return "The model returned an empty response."
                    logger.debug("No standard fields found in JSON, returning stringified dict")
                    return json.dumps(json_obj, indent=2) # Return formatted JSON string
                else:
                     # If parsed JSON is not a dict (e.g., list, string), return original text
                     logger.debug("Parsed JSON is not a dictionary, returning original text")
                     return response_text
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed ({e}), returning original text")
                return response_text # Not valid JSON, return as is
        else:
            # Doesn't look like JSON, return original text
            return response_text

class OpenAIEngine(InferenceEngine):
    """OpenAI API inference engine implementation"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it.")
        try:
            # Defer connection/model check until first use if desired,
            # but checking at init provides earlier feedback.
            self._check_connection()
            self._check_model()
        except Exception as e:
            logger.warning(f"OpenAI initialization issue: {str(e)}")

    @staticmethod
    def get_available_models() -> list:
        """Get list of available OpenAI models"""
        return [
            "gpt-3.5-turbo",
            "gpt-4",        # Standard GPT-4
            "gpt-4-turbo",  # Latest Turbo model
            "gpt-4o",       # Latest Omni model
            # Vision models are typically handled implicitly when image data is sent
            # "gpt-4-vision-preview" # Example specific vision model if needed
        ]

    def _check_connection(self):
        """Check if the API key is valid and OpenAI is reachable"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "model": "gpt-3.5-turbo", # Use a cheap model for the check
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1, # Minimize cost/time
                    "temperature": 0
                },
                timeout=10 # Reasonable timeout for a ping
            )
            # Allow 400 Bad Request (e.g., if 'ping' is invalid content) but not auth errors (401) or server errors (5xx)
            if response.status_code == 401:
                 raise ConnectionError(f"OpenAI API key is invalid (status {response.status_code})")
            elif response.status_code >= 500:
                 raise ConnectionError(f"OpenAI server error (status {response.status_code})")
            elif response.status_code not in [200, 400]:
                 # Log other client errors but don't necessarily raise ConnectionError unless critical
                 logger.warning(f"Unexpected status during OpenAI connection check: {response.status_code} - {response.text}")
            logger.info("OpenAI connection check successful.")
        except requests.RequestException as e:
            raise ConnectionError(f"Network error connecting to OpenAI API: {str(e)}")
        except Exception as e: # Catch other potential errors during check
             logger.error(f"Unexpected error during OpenAI connection check: {e}")
             raise ConnectionError(f"Unexpected error connecting to OpenAI API: {str(e)}")

    def _check_model(self):
        """Check if the requested model is in the known list (basic check)."""
        # Note: A more robust check would involve listing models via the API,
        # but that adds complexity and another API call.
        available = self.get_available_models()
        if self.model_name not in available:
            logger.warning(f"Model '{self.model_name}' is not in the hardcoded known OpenAI model list: {available}")
            # Allow using potentially newer models not in the list, but log a warning.
            # Return True to allow the attempt, the API call will fail if invalid.
            # return False # Uncomment this to strictly enforce the list
        return True

    def generate(self, prompt: str, params: Dict[Any, Any], stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate response from OpenAI, supporting streaming."""
        if not self._check_model():
             error_msg = f"Error: Model '{self.model_name}' is not in the known OpenAI model list (or check failed)."
             if stream:
                 def error_gen(): yield error_msg
                 return error_gen()
             return error_msg
        try:

            temperature = params.get('creativity', 0.5)
            try:
                temperature = float(temperature)
            except (ValueError, TypeError):
                temperature = 0.5

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            image_data_url = params.get("image_base64")

            if image_data_url:
                # Vision prompt with image
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }]
            else:
                # Standard text-only input
                messages = []
                system_prompt = params.get("system_prompt")
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": stream # Use the stream parameter
            }

            logger.debug(f"Sending request to OpenAI (model: {self.model_name}, stream: {stream}) with payload:\n{json.dumps(payload, indent=2)}")

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=stream, # Pass stream argument to requests
                timeout=180 # Longer timeout for potentially long generations
            )
            logger.debug(f"OpenAI response status: {response.status_code}")
            response.raise_for_status() # Raise HTTPError for bad responses

            if stream:
                return self._process_openai_stream(response)
            else:
                result = response.json()
                logger.debug(f"OpenAI non-stream response received: {str(result)[:200]}...")
                if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
                    content = result['choices'][0]['message']['content']
                    logger.debug(f"OpenAI response content (truncated): {content[:100]}...")
                    return content
                else:
                    logger.error(f"Unexpected OpenAI response format (non-stream): {result}")
                    return "Error: Unexpected response format from OpenAI"

        except requests.HTTPError as e:
            error_msg = f"OpenAI HTTP error ({e.response.status_code})"
            try:
                error_detail = e.response.json().get('error', {}).get('message', str(e))
                error_msg += f": {error_detail}"
            except json.JSONDecodeError:
                 error_msg += f": {e.response.text}"
            logger.error(error_msg)
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg
        except requests.RequestException as e:
            error_msg = f"Network error connecting to OpenAI: {str(e)}"
            logger.error(error_msg)
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred during OpenAI generation: {str(e)}"
            logger.exception(error_msg)
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg

    def _process_openai_stream(self, response: requests.Response) -> Generator[str, None, None]:
        """Processes the streaming response (Server-Sent Events) from OpenAI."""
        try:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[len("data: "):]
                        if data_str.strip() == "[DONE]":
                            logger.debug("OpenAI stream finished.")
                            break # Stream finished
                        try:
                            data = json.loads(data_str)
                            if data.get('choices') and data['choices'][0].get('delta') and 'content' in data['choices'][0]['delta']:
                                chunk = data['choices'][0]['delta']['content']
                                if chunk: # Ensure content is not None or empty
                                    yield chunk
                        except json.JSONDecodeError:
                            logger.error(f"Error decoding JSON from OpenAI stream: {data_str}")
                            yield f"\nError decoding stream data\n"
                        except Exception as e:
                             logger.error(f"Error processing OpenAI stream data chunk: {e} - Data: {data_str}")
                             yield f"\nError processing stream chunk: {e}\n"
        except requests.exceptions.ChunkedEncodingError as e:
            logger.error(f"OpenAI stream connection error: {e}")
            yield f"\nStream connection error: {str(e)}\n"
        except Exception as e:
            logger.exception(f"Unexpected error during OpenAI stream processing")
            yield f"\nUnexpected stream error: {str(e)}\n"


class ClaudeEngine(InferenceEngine):
    """Claude API inference engine implementation"""

    def __init__(self, model_name: str = "claude-3-haiku-20240307"): # Default to a common modern model
        self.model_name = model_name
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set. Please set it.")
        # Add connection/model check if desired
        # try:
        #     self._check_connection()
        # except Exception as e:
        #     logger.warning(f"Claude initialization issue: {str(e)}")

    @staticmethod
    @staticmethod
    def get_available_models() -> list:
        """Get list of common Claude models (as of early 2024)."""
        # See https://docs.anthropic.com/claude/docs/models-overview
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]

    # Optional: Add _check_connection and _check_model similar to OpenAI

    def generate(self, prompt: str, params: Dict[Any, Any], stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate response from Claude, supporting streaming."""
        try:
            temperature = params.get('creativity', 0.5)
            try:
                temperature = float(temperature)
            except (ValueError, TypeError):
                temperature = 0.5

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01", # Required API version
                "content-type": "application/json"
            }

            # Handle image data if present (Claude specific format)
            image_data_url = params.get("image_base64")
            messages = []
            system_prompt = params.get("system_prompt")
            if system_prompt:
                 messages.append({"role": "system", "content": system_prompt}) # Claude uses 'system' role differently

            user_content = []
            user_content.append({"type": "text", "text": prompt})

            if image_data_url:
                 # Extract mime type and base64 data
                 try:
                     header, encoded = image_data_url.split(",", 1)
                     mime_type = header.split(":")[1].split(";")[0]
                     # Ensure supported image type
                     if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                          raise ValueError(f"Unsupported image type for Claude: {mime_type}")
                     user_content.append({
                         "type": "image",
                         "source": {
                             "type": "base64",
                             "media_type": mime_type,
                             "data": encoded,
                         },
                     })
                 except Exception as img_e:
                     logger.error(f"Error processing image data for Claude: {img_e}")
                     # Decide whether to proceed without image or raise error
                     # For now, proceed without image but log error
                     pass # Or: return "Error processing image data"

            messages.append({"role": "user", "content": user_content})


            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
                "max_tokens": 4096 # Set a reasonable max token limit
            }

            logger.debug(f"Sending request to Claude (model: {self.model_name}, stream: {stream}) with payload:\n{json.dumps(payload, indent=2)}")

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=stream,
                timeout=180
            )
            logger.debug(f"Claude response status: {response.status_code}")
            response.raise_for_status()

            if stream:
                return self._process_claude_stream(response)
            else:
                result = response.json()
                logger.debug(f"Claude non-stream response received: {str(result)[:200]}...")
                if result.get('content') and isinstance(result['content'], list) and result['content'][0].get('text'):
                    return result['content'][0]['text']
                else:
                    logger.error(f"Unexpected Claude response format (non-stream): {result}")
                    # Check for error structure
                    if result.get('type') == 'error' and result.get('error'):
                        return f"Claude API Error: {result['error'].get('type')} - {result['error'].get('message', '')}"
                    return "Error: Unexpected response format from Claude"

        except requests.HTTPError as e:
            error_msg = f"Claude HTTP error ({e.response.status_code})"
            try:
                error_detail = e.response.json()
                if error_detail.get('type') == 'error' and error_detail.get('error'):
                     error_msg += f": {error_detail['error'].get('type')} - {error_detail['error'].get('message', str(e))}"
                else:
                     error_msg += f": {e.response.text}"
            except json.JSONDecodeError:
                 error_msg += f": {e.response.text}"
            logger.error(error_msg)
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg
        except requests.RequestException as e:
            error_msg = f"Network error connecting to Claude: {str(e)}"
            logger.error(error_msg)
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred during Claude generation: {str(e)}"
            logger.exception(error_msg)
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg


    def _process_claude_stream(self, response: requests.Response) -> Generator[str, None, None]:
        """Processes the streaming response (Server-Sent Events) from Claude."""
        try:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("event: "):
                        event_type = decoded_line[len("event: "):].strip()
                    elif decoded_line.startswith("data: "):
                        data_str = decoded_line[len("data: "):]
                        try:
                            data = json.loads(data_str)
                            event_type = data.get('type') # Get type from data as well

                            if event_type == "content_block_delta" and data.get('delta', {}).get('type') == 'text_delta':
                                yield data['delta']['text']
                            elif event_type == "message_stop":
                                logger.debug("Claude stream finished.")
                                break
                            elif event_type == "error":
                                error_data = data.get('error', {})
                                error_msg = f"Claude API Error: {error_data.get('type')} - {error_data.get('message', 'Unknown error')}"
                                logger.error(error_msg)
                                yield f"\n{error_msg}\n"
                                break # Stop streaming on error
                            # Handle other event types if needed (e.g., message_start, content_block_start/stop)
                            # else:
                            #     logger.debug(f"Received Claude stream event: {event_type}")

                        except json.JSONDecodeError:
                            logger.error(f"Error decoding JSON from Claude stream: {data_str}")
                            yield "\nError decoding stream data\n"
                        except Exception as e:
                             logger.error(f"Error processing Claude stream data chunk: {e} - Data: {data_str}")
                             yield f"\nError processing stream chunk: {e}\n"
        except requests.exceptions.ChunkedEncodingError as e:
            logger.error(f"Claude stream connection error: {e}")
            yield f"\nStream connection error: {str(e)}\n"
        except Exception as e:
            logger.exception(f"Unexpected error during Claude stream processing")
            yield f"\nUnexpected stream error: {str(e)}\n"
