import streamlit as st
import base64
import time # For potential delays or UI updates
import os # For API keys
import logging
from src.preprocessing.processor import PromptProcessor
from src.inferencing.inference import InferenceEngine, OllamaEngine, ClaudeEngine, OpenAIEngine

logger = logging.getLogger(__name__)


def create_interface(processor: PromptProcessor):
    # Set page config only once, at the start
    st.set_page_config(layout="wide", page_title="Prism AI")

    st.title("Prism AI")
    st.caption("LLM Alignment and Prompt Processing Interface")

    # --- Session State Initialization ---
    # Initialize necessary keys in session state if they don't exist
    # Use more descriptive keys
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_alignment_text" not in st.session_state:
        st.session_state.current_alignment_text = "" # Store the input alignment text
    if "current_alignment_image_url" not in st.session_state:
        st.session_state.current_alignment_image_url = None # Store the encoded image data
    if "alignment_response_complete" not in st.session_state:
        st.session_state.alignment_response_complete = "" # Store full response after streaming
    if "main_response_complete" not in st.session_state:
        st.session_state.main_response_complete = "" # Store full response after streaming
    if "no_alignment_response_complete" not in st.session_state:
        st.session_state.no_alignment_response_complete = "" # Store full response after streaming
    if "alignment_engine_type" not in st.session_state:
        st.session_state.alignment_engine_type = "ollama" # Default
    if "alignment_model_name" not in st.session_state:
        st.session_state.alignment_model_name = "llama3" # Default
    if "main_engine_type" not in st.session_state:
        st.session_state.main_engine_type = "ollama" # Default
    if "main_model_name" not in st.session_state:
        st.session_state.main_model_name = "llama3" # Default


    # --- Helper Functions ---
    def get_models_for_engine(engine_type: str) -> list:
        """Safely gets models for a given engine type."""
        try:
            if engine_type == "ollama":
                models = OllamaEngine.get_available_models()
                if not models:
                    st.warning("No Ollama models found. Is Ollama running and models pulled?")
                    return ["(Ollama not reachable or no models)"]
                return models
            elif engine_type == "claude":
                # Check for API key before listing models
                if not os.getenv("ANTHROPIC_API_KEY"):
                     st.warning("Claude engine selected, but ANTHROPIC_API_KEY is not set.")
                     return ["(Set API Key first)"]
                return ClaudeEngine.get_available_models()
            elif engine_type == "openai":
                 if not os.getenv("OPENAI_API_KEY"):
                     st.warning("OpenAI engine selected, but OPENAI_API_KEY is not set.")
                     return ["(Set API Key first)"]
                 return OpenAIEngine.get_available_models()
            else:
                return ["(unknown engine)"]
        except Exception as e:
            st.error(f"Error getting models for {engine_type}: {str(e)}")
            logger.error(f"Error getting models for {engine_type}", exc_info=True)
            return ["(error loading models)"]

    def encode_image_to_base64(uploaded_file):
        """Encodes uploaded image file to base64 data URL."""
        if uploaded_file is None:
            return None
        try:
            image_bytes = uploaded_file.read()
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = uploaded_file.type or "image/jpeg" # Default mime type
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            st.error(f"Error encoding image: {e}")
            logger.error("Error encoding image", exc_info=True)
            return None

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        # --- Alignment Model Config ---
        with st.expander("Alignment Model", expanded=True):
            available_engines = InferenceEngine.get_available_engines()

            # Use session state for engine selection persistence
            st.session_state.alignment_engine_type = st.selectbox(
                "Alignment Engine",
                available_engines,
                index=available_engines.index(st.session_state.alignment_engine_type) if st.session_state.alignment_engine_type in available_engines else 0,
                key="sb_align_engine" # Unique key
            )
            align_engine_type = st.session_state.alignment_engine_type # Get current value

            # Get models based on selected engine
            align_models = get_models_for_engine(align_engine_type)
            # Try to find the index of the currently selected model
            try:
                align_model_index = align_models.index(st.session_state.alignment_model_name)
            except ValueError:
                 align_model_index = 0 # Default to first if not found or list changed

            st.session_state.alignment_model_name = st.selectbox(
                "Alignment Model",
                align_models,
                index=align_model_index if align_models else 0, # Handle empty list case
                key="sb_align_model"
            )
            align_model = st.session_state.alignment_model_name # Get current value

            # Custom model option (less critical now with dynamic lists, but keep for flexibility)
            # use_custom_align_model = st.checkbox("Use custom alignment model name", key="cb_align_custom")
            # custom_align_model = st.text_input(
            #     "Custom alignment model",
            #     value=st.session_state.alignment_model_name if use_custom_align_model else "",
            #     disabled=not use_custom_align_model,
            #     key="ti_align_custom"
            # )
            # actual_align_model = custom_align_model if use_custom_align_model and custom_align_model else align_model

            # API Key inputs - Placed logically with engine selection
            # Use os.getenv to check if already set, avoid overwriting unless new input provided
            if align_engine_type == "openai":
                openai_key_current = os.getenv("OPENAI_API_KEY", "")
                openai_api_key_input = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="Enter your OpenAI API Key" if not openai_key_current else "********",
                    key="ti_openai_key"
                )
                if openai_api_key_input: # Only set if user provides new input
                    os.environ["OPENAI_API_KEY"] = openai_api_key_input
                    st.success("OpenAI API Key set for this session.", icon="🔑")


            elif align_engine_type == "claude":
                claude_key_current = os.getenv("ANTHROPIC_API_KEY", "")
                claude_api_key_input = st.text_input(
                    "Claude API Key",
                    type="password",
                    placeholder="Enter your Anthropic API Key" if not claude_key_current else "********",
                    key="ti_claude_key"
                )
                if claude_api_key_input:
                    os.environ["ANTHROPIC_API_KEY"] = claude_api_key_input
                    st.success("Claude API Key set for this session.", icon="🔑")
                    
        # --- Main Model Config ---
        with st.expander("Main Processing Model", expanded=True):
             # Use session state for persistence
            st.session_state.main_engine_type = st.selectbox(
                "Main Engine",
                available_engines,
                index=available_engines.index(st.session_state.main_engine_type) if st.session_state.main_engine_type in available_engines else 0,
                key="sb_main_engine"
            )
            main_engine_type = st.session_state.main_engine_type # Get current value

            # API Key inputs (show if needed for main engine, even if different from alignment)
            if main_engine_type == "openai" and align_engine_type != "openai":
                openai_key_current = os.getenv("OPENAI_API_KEY", "")
                openai_api_key_input = st.text_input(
                    "OpenAI API Key (for Main)",
                    type="password",
                    placeholder="Enter your OpenAI API Key" if not openai_key_current else "********",
                    key="ti_openai_key_main" # Different key
                )
                if openai_api_key_input:
                    os.environ["OPENAI_API_KEY"] = openai_api_key_input
                    st.success("OpenAI API Key set for this session.", icon="🔑")
            elif main_engine_type == "claude" and align_engine_type != "claude":
                claude_key_current = os.getenv("ANTHROPIC_API_KEY", "")
                claude_api_key_input = st.text_input(
                    "Claude API Key (for Main)",
                    type="password",
                    placeholder="Enter your Anthropic API Key" if not claude_key_current else "********",
                    key="ti_claude_key_main" # Different key
                )
                if claude_api_key_input:
                    os.environ["ANTHROPIC_API_KEY"] = claude_api_key_input
                    st.success("Claude API Key set for this session.", icon="🔑")


            main_models = get_models_for_engine(main_engine_type)
            try:
                 main_model_index = main_models.index(st.session_state.main_model_name)
            except ValueError:
                 main_model_index = 0

            st.session_state.main_model_name = st.selectbox(
                "Main Model",
                main_models,
                index=main_model_index if main_models else 0,
                key="sb_main_model"
            )
            main_model = st.session_state.main_model_name # Get current value

            # Custom model option (Main)
            # use_custom_main_model = st.checkbox("Use custom main model name", key="cb_main_custom")
            # custom_main_model = st.text_input(
            #     "Custom main model",
            #     value=st.session_state.main_model_name if use_custom_main_model else "",
            #     disabled=not use_custom_main_model,
            #     key="ti_main_custom"
            # )
            # actual_main_model = custom_main_model if use_custom_main_model and custom_main_model else main_model


        # --- Processing Parameters ---
        with st.expander("Processing Parameters", expanded=False):
             # Style selection (can influence alignment and main response)
            style = st.radio(
                "Writing Style",
                ["Casual", "Professional", "Academic"],
                index=1, # Default to Professional
                key="rad_style"
            )

            # Tone selection
            tone = st.selectbox(
                "Tone",
                ["Neutral", "Positive", "Negative"],
                index=0, # Default to Neutral
                key="sb_tone"
            )

            # Creativity slider (Temperature)
            creativity = st.slider(
                "Creativity (Temperature)",
                min_value=0.0,
                max_value=1.5, # Allow higher values
                value=0.5,
                step=0.1,
                key="sl_creativity"
            )

        # --- History Section ---
        with st.expander("History", expanded=False):
            if st.button("Clear History", key="btn_clear_history"):
                st.session_state.chat_history = []
                st.rerun() # Update display immediately

            if st.session_state.chat_history:
                # Display history entries
                st.write(f"{len(st.session_state.chat_history)} entries in history.")
                # Show last 5-10 entries without nested expanders for better readability
                for i, entry in enumerate(reversed(st.session_state.chat_history[-10:])):
                    entry_num = len(st.session_state.chat_history) - i
                    with st.container(border=True):
                         st.markdown(f"**Entry {entry_num}**")
                         if entry.get('alignment_input'):
                             st.markdown(f"**Alignment Input:**\n```\n{entry['alignment_input']}\n```")
                         if entry.get('alignment_response'):
                             st.markdown(f"**Alignment Response:**\n```\n{entry['alignment_response']}\n```")
                         st.markdown(f"**Main Prompt:**\n```\n{entry['prompt']}\n```")
                         if entry.get('response_with_alignment'):
                             st.markdown(f"**Response (with alignment):**\n```\n{entry['response_with_alignment']}\n```")
                         if entry.get('response_without_alignment'):
                             st.markdown(f"**Response (without alignment):**\n```\n{entry['response_without_alignment']}\n```")
                    # st.divider() # Use container border instead
            else:
                st.info("No history yet.")

    # --- Parameter Dictionary (Constructed just before use) ---
    # Moved construction closer to where params are used

    # --- Main Content Area ---

    col1, col2 = st.columns(2, gap="medium")

    # --- Column 1: Alignment ---
    with col1:
        st.header("1. Alignment Input")
        st.caption("Provide text or an image to guide the model's behavior.")

        # Alignment text input - Use session state to preserve input across reruns
        st.session_state.current_alignment_text = st.text_area(
            "Alignment Text",
            value=st.session_state.current_alignment_text, # Bind to session state
            placeholder="Enter text to align the model (e.g., principles, style guide, persona description)...",
            height=150,
            key="ta_align_text"
        )
        alignment_text = st.session_state.current_alignment_text # Get current value

        # Image uploader
        uploaded_image = st.file_uploader(
            "Upload Alignment Image (Optional)",
            type=["jpg", "jpeg", "png", "gif", "webp"], # Supported types
            key="fu_align_image"
        )

        # Display uploaded image thumbnail and store encoded data
        if uploaded_image:
            st.image(uploaded_image, width=100)
            st.session_state.current_alignment_image_url = encode_image_to_base64(uploaded_image)
        else:
            # Clear image data if uploader is cleared
            st.session_state.current_alignment_image_url = None
            
        # URL input for content extraction
        alignment_url = st.text_input(
            "Or enter URL for content extraction",
            placeholder="https://example.com/article",
            help="Enter a URL to extract content for alignment. The main article content will be extracted automatically. For best results, install trafilatura package.",
            key="ti_align_url"
        )

        # Alignment action buttons
        align_btn_col, reset_btn_col = st.columns(2)
        with align_btn_col:
            align_button_pressed = st.button("Generate Alignment Guidance", type="primary", key="btn_align")
        with reset_btn_col:
            if st.button("Reset Alignment Input", key="btn_reset_align"):
                 st.session_state.current_alignment_text = ""
                 st.session_state.current_alignment_image_url = None
                 # Also clear the URL input field
                 st.session_state["ti_align_url"] = ""
                 # Clear the file uploader state requires a rerun trick or specific component handling
                 # For now, just clear the text and stored data. User needs to manually clear uploader.
                 st.session_state.alignment_response_complete = "" # Also clear previous output
                 st.rerun()


        # Alignment output area (placeholder for streaming)
        st.subheader("Alignment Guidance Output")
        st.caption("The model's interpretation of your alignment input.")
        alignment_output_placeholder = st.empty() # Placeholder for streaming content
        # Display the completed response from session state if not currently streaming
        if not align_button_pressed:
             alignment_output_placeholder.markdown(st.session_state.alignment_response_complete, unsafe_allow_html=True)


        # --- Alignment Processing Logic ---
        if align_button_pressed:
            # Ensure either text, image, or URL is provided
            if not alignment_text and not st.session_state.current_alignment_image_url and not alignment_url:
                st.warning("Please provide alignment text, upload an image, or enter a URL.")
            else:
                # Construct params for alignment
                alignment_params = {
                    "style": style, # Use sidebar params
                    "tone": tone,
                    "creativity": float(creativity),
                    "image_base64": st.session_state.current_alignment_image_url,
                    "alignment_url": alignment_url # Pass the URL for extraction
                    # 'format' is usually not needed/handled by alignment prompt
                }
                
                # If URL is provided, show a warning about trafilatura if needed
                if alignment_url:
                    try:
                        import trafilatura
                    except ImportError:
                        st.warning("The trafilatura package is not installed. A basic fallback extraction method will be used. For better results, install trafilatura with: pip install trafilatura")
                    
                    # Show a loading indicator when extracting from URL
                    st.info(f"Extracting content from {alignment_url}...")

                # Get selected alignment engine/model
                align_engine = st.session_state.alignment_engine_type
                align_model = st.session_state.alignment_model_name
                # actual_align_model = custom_align_model if use_custom_align_model and custom_align_model else align_model

                # Display info message first
                st.info(f"Generating alignment guidance using: {align_engine} / {align_model}")
                # Then set the placeholder message
                alignment_output_placeholder.markdown("Generating...") # Initial message

                try:
                    # Ensure the processor's alignment engine is up-to-date
                    # This might fail if API keys are missing etc.
                    processor.alignment_engine = InferenceEngine.create_engine(align_engine, align_model)

                    # Call the streaming processor method
                    response_stream = processor.process_alignment(alignment_text, alignment_params)

                    # Use st.write_stream to display the output
                    # Capture the full response after streaming for session state
                    full_response = alignment_output_placeholder.write_stream(response_stream)
                    st.session_state.alignment_response_complete = full_response
                    logger.info("Alignment guidance generated successfully.")
                    # Optional: Rerun if needed, but write_stream updates live
                    # st.rerun()

                except ValueError as ve:
                     st.error(f"Configuration Error: {ve}")
                     logger.error(f"Alignment config error: {ve}")
                     alignment_output_placeholder.error(f"Configuration Error: {ve}")
                     st.session_state.alignment_response_complete = f"**Error:** {ve}"
                except ConnectionError as ce:
                     st.error(f"Connection Error: Could not connect to {align_engine}. Is it running? {ce}")
                     logger.error(f"Alignment connection error: {ce}")
                     alignment_output_placeholder.error(f"Connection Error: {ce}")
                     st.session_state.alignment_response_complete = f"**Error:** {ce}"
                except Exception as e:
                    st.error(f"Error during alignment processing: {str(e)}")
                    logger.error("Alignment processing error", exc_info=True)
                    alignment_output_placeholder.error(f"An unexpected error occurred: {str(e)}")
                    st.session_state.alignment_response_complete = f"**Error:** {str(e)}"

    # --- Column 2: Main Prompt and Output ---
    with col2:
        st.header("2. Main Prompt")
        st.caption("Enter your main request for the model.")

        # Main prompt input
        main_prompt_input = st.text_area(
            "Main Prompt",
            placeholder="Enter your prompt here...",
            height=150,
            key="ta_main_prompt"
        )

        # Process button
        process_button_pressed = st.button("Generate Response", type="primary", key="btn_process")

        # --- Main Processing Logic ---
        if process_button_pressed:
            if not main_prompt_input:
                st.warning("Please enter a main prompt.")
            else:
                # Construct params for main processing
                main_params = {
                    "style": style,
                    "tone": tone,
                    "creativity": float(creativity),
                    # 'format': 'json' # Only add if main model should output JSON
                    # Image data is usually handled in alignment, not passed directly here
                }

                # Get selected main engine/model
                main_engine = st.session_state.main_engine_type
                main_model = st.session_state.main_model_name
                # actual_main_model = custom_main_model if use_custom_main_model and custom_main_model else main_model

                st.info(f"Processing prompt using: {main_engine} / {main_model}")

                # --- Response WITH Alignment ---
                st.subheader("Response (with Alignment)")
                with_align_output_placeholder = st.empty()
                with_align_output_placeholder.markdown("Generating...")

                # --- Response WITHOUT Alignment ---
                st.subheader("Response (without Alignment)")
                no_align_output_placeholder = st.empty()
                no_align_output_placeholder.markdown("Generating...")

                # Use placeholders within columns for better layout control during streaming
                # response_col1, response_col2 = st.columns(2)

                try:
                    # Set the main engine in the processor
                    processor.set_main_engine(main_engine, main_model)

                    # --- Generate WITH alignment ---
                    logger.info("Generating response WITH alignment...")
                    # Check if alignment result contains an error
                    alignment_result = st.session_state.alignment_response_complete
                    if alignment_result and alignment_result.startswith("Error:"):
                        logger.warning(f"Using alignment with error: {alignment_result}")
                        # We'll still pass it to process_main, which will handle the error appropriately
                    
                    response_stream_with_align = processor.process_main(
                        prompt=main_prompt_input,
                        alignment_result=alignment_result,
                        params=main_params
                    )
                    # Stream response WITH alignment
                    full_response_with_align = with_align_output_placeholder.write_stream(response_stream_with_align)
                    st.session_state.main_response_complete = full_response_with_align
                    logger.info("Response WITH alignment generated.")

                    # --- Generate WITHOUT alignment ---
                    # Use the main engine directly, bypassing alignment-specific logic in processor.process_main
                    logger.info("Generating response WITHOUT alignment...")
                    no_align_params = main_params.copy()
                    no_align_params['system_prompt'] = "" # No system prompt derived from alignment
                    no_align_params['is_alignment'] = False

                    # Ensure main_engine is available (set_main_engine should have run)
                    if processor.main_engine:
                        response_stream_no_align = processor.main_engine.generate(
                            prompt=main_prompt_input.strip(), # Use raw prompt
                            params=no_align_params,
                            stream=True
                        )
                        # Stream response WITHOUT alignment
                        full_response_no_align = no_align_output_placeholder.write_stream(response_stream_no_align)
                        st.session_state.no_alignment_response_complete = full_response_no_align
                        logger.info("Response WITHOUT alignment generated.")
                    else:
                         st.session_state.no_alignment_response_complete = "Error: Main engine not available for non-aligned generation."
                         no_align_output_placeholder.error(st.session_state.no_alignment_response_complete)


                    # --- Add to History ---
                    history_entry = {
                        "timestamp": time.time(),
                        "alignment_input": st.session_state.current_alignment_text,
                        # Add image identifier if needed: "alignment_image": uploaded_image.name if uploaded_image else None,
                        "alignment_response": st.session_state.alignment_response_complete,
                        "prompt": main_prompt_input,
                        "response_with_alignment": st.session_state.main_response_complete,
                        "response_without_alignment": st.session_state.no_alignment_response_complete,
                        "params": { # Store params used for this generation
                             "align_engine": st.session_state.alignment_engine_type, # Read from session state
                             "align_model": st.session_state.alignment_model_name,   # Read from session state
                             "main_engine": main_engine, "main_model": main_model,
                             "style": style, "tone": tone, "creativity": creativity
                        }
                    }
                    st.session_state.chat_history.append(history_entry)
                    logger.info("Entry added to chat history.")
                    # No rerun needed here, streaming updated the UI

                except ValueError as ve:
                     st.error(f"Configuration Error: {ve}")
                     logger.error(f"Main processing config error: {ve}")
                     with_align_output_placeholder.error(f"Configuration Error: {ve}")
                     no_align_output_placeholder.empty() # Clear the other placeholder
                     st.session_state.main_response_complete = f"**Error:** {ve}"
                     st.session_state.no_alignment_response_complete = ""
                except ConnectionError as ce:
                     st.error(f"Connection Error: Could not connect to {main_engine}. Is it running? {ce}")
                     logger.error(f"Main processing connection error: {ce}")
                     with_align_output_placeholder.error(f"Connection Error: {ce}")
                     no_align_output_placeholder.empty()
                     st.session_state.main_response_complete = f"**Error:** {ce}"
                     st.session_state.no_alignment_response_complete = ""
                except Exception as e:
                    st.error(f"Error during main processing: {str(e)}")
                    logger.error("Main processing error", exc_info=True)
                    with_align_output_placeholder.error(f"An unexpected error occurred: {str(e)}")
                    no_align_output_placeholder.empty()
                    st.session_state.main_response_complete = f"**Error:** {str(e)}"
                    st.session_state.no_alignment_response_complete = ""

        # --- Display Areas (Static/Completed) ---
        # These areas are no longer needed as the placeholders above handle the full display via write_stream.
        # The placeholders will show the final content once streaming is complete.


    # --- Copy Buttons (Placed at the bottom or in expanders) ---
    st.divider()
    copy_col1, copy_col2, copy_col3 = st.columns(3)

    with copy_col1:
        if st.session_state.alignment_response_complete:
            st.download_button(
                label="Copy Alignment Guidance",
                data=st.session_state.alignment_response_complete,
                file_name="alignment_guidance.txt",
                mime="text/plain",
                key="btn_copy_align"
            )

    with copy_col2:
        if st.session_state.main_response_complete:
            st.download_button(
                label="Copy Response (with Alignment)",
                data=st.session_state.main_response_complete,
                file_name="response_with_alignment.txt",
                mime="text/plain",
                key="btn_copy_main"
            )

    with copy_col3:
        if st.session_state.no_alignment_response_complete:
            st.download_button(
                label="Copy Response (without Alignment)",
                data=st.session_state.no_alignment_response_complete,
                file_name="response_without_alignment.txt",
                mime="text/plain",
                key="btn_copy_no_align"
            )

    # No explicit return needed for Streamlit script
