import streamlit as st
import base64
from src.preprocessing.processor import PromptProcessor
from src.inferencing.inference import InferenceEngine, OllamaEngine, ClaudeEngine, OpenAIEngine


def create_interface(processor: PromptProcessor):
    # Set page layout to wide
    st.set_page_config(layout="wide")
    
    st.title("Prism AI")

    # Initialize history in session state if not already present
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar for model selection and parameters
    with st.sidebar:
        with st.expander("Model Configuration (Alignment)", expanded=False):
            # Engine selection
            available_engines = InferenceEngine.get_available_engines()
            engine_type = st.selectbox(
                "Select Engine",
                available_engines,
                index=0

            )

            # Model selection
            def get_models_for_engine(engine_type: str) -> list:
                try:
                    if engine_type == "ollama":
                        models = OllamaEngine.get_available_models()
                        if not models:
                            st.warning("No Ollama models found. Is Ollama running?")
                            return ["(none available - is Ollama running?)"]
                        return models
                    elif engine_type == "claude":
                        return ClaudeEngine.get_available_models()
                    elif engine_type == "openai":
                        return OpenAIEngine.get_available_models()
                    else:
                        return ["(unknown engine)"]
                except Exception as e:
                    st.error(f"Error getting models for {engine_type}: {str(e)}")
                    return ["(error loading models)"]

            models = get_models_for_engine(engine_type)
            model = st.selectbox(
                "Select Model",
                models,
                index=0 if models else None
            )

            # Custom model option
            use_custom_model = st.checkbox("Use custom model name")
            custom_model = st.text_input(
                "Custom model name",
                disabled=not use_custom_model
            )

            # API Key inputs (only shown for relevant engines)
            if engine_type == "openai":
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Enter your OpenAI API Key"
                )
                if api_key:
                    import os
                    os.environ["OPENAI_API_KEY"] = api_key
            
            elif engine_type == "claude":
                claude_api_key = st.text_input(
                    "Claude API Key",
                    type="password",
                    help="Enter your Anthropic Claude API Key"
                )
                if claude_api_key:
                    import os
                    os.environ["ANTHROPIC_API_KEY"] = claude_api_key
                    
            # Style selection
            style = st.radio(
                "Writing Style",
                ["Casual", "Professional", "Academic"],
                index=1
            )

            # Tone selection
            tone = st.selectbox(
                "Tone",
                ["Neutral", "Positive", "Negative"],
                index=0
            )

            # Creativity slider
            creativity = st.slider(
                "Creativity Level",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )

        with st.expander("Main Model Configuration", expanded=False):
            # Main engine selection
            main_engine_type = st.selectbox(
                "Main Processing Engine",
                available_engines,
                index=0,
                key="main_engine"
            )

            # Main model selection
            main_models = get_models_for_engine(main_engine_type)
            main_model = st.selectbox(
                "Main Processing Model",
                main_models,
                index=0 if main_models else None,
                key="main_model"
            )

            # Main custom model option
            main_use_custom_model = st.checkbox("Use custom main model name", key="main_custom")
            main_custom_model = st.text_input(
                "Custom main model name",
                disabled=not main_use_custom_model,
                key="main_custom_input"
            )
            
        with st.expander("Processing Parameters", expanded=False):
            # Empty section - parameters moved to Model Configuration
            pass
        # History Section in Sidebar
        with st.expander("History", expanded=False):
            if st.button("Clear History", key="clear_history_sidebar"):
                st.session_state.history = []

            if st.session_state.get("history"):
                for i, entry in enumerate(reversed(st.session_state.history[-10:]), 1):
                    with st.expander(f"Prompt: {entry['prompt'][:30]}...", expanded=False):
                        st.markdown(f"**Prompt:**\n{entry['prompt']}")
                        st.markdown(f"**Response:**\n{entry['response']}")
            else:
                st.info("No history yet. Process a prompt to see it here.")

        # Parameters dictionary
        params = {
            "style": style,
            "tone": tone,
            "creativity": float(creativity),
            "format": "json"
        }

    # Initialize session state for responses if not already present
    if 'alignment_response' not in st.session_state:
        st.session_state.alignment_response = ""
    if 'final_output' not in st.session_state:
        st.session_state.final_output = ""

    # Main content area - create two columns for side-by-side layout with full width
    # Use a wider layout with minimal gap to maximize usable space
    left_col, right_col = st.columns([1, 1], gap="small")
    
    # Left column - Alignment section
    with left_col:
        st.header("Alignment")

        # Alignment text input
        alignment_text = st.text_area(
            "Add your alignment text here",
            placeholder="Enter text to align the model's behavior...",
            height=150
        )

        uploaded_image = st.file_uploader("Or upload an image to help align the model", type=["jpg", "jpeg", "png"])

        def encode_image_to_base64(uploaded_file):
            if uploaded_file is None:
                return None
            image_bytes = uploaded_file.read()
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = uploaded_file.type or "image/jpeg"
            return f"data:{mime_type};base64,{encoded}"

        image_data_url = encode_image_to_base64(uploaded_image)
        if image_data_url:
            params["image_base64"] = image_data_url

        # Alignment buttons
        align_col1, align_col2, align_col3 = st.columns(3)
        with align_col1:
            align_btn = st.button("Align", type="primary")
        with align_col2:
            align_interrupt_btn = st.button("Interrupt", type="secondary")
        with align_col3:
            align_reset_btn = st.button("Reset", type="secondary")
        
        # Alignment output - always visible
        st.subheader("Model response to alignment")
        alignment_output_container = st.container(height=200, border=True)
        with alignment_output_container:
            st.markdown(st.session_state.alignment_response)
        
        # Process alignment when button is clicked
        if align_btn and alignment_text:
            try:
                # Use custom model name if checkbox is checked
                actual_model = custom_model if use_custom_model and custom_model else model
                st.info(f"Using alignment model: {actual_model} with engine: {engine_type}")

                # Set the alignment engine explicitly before processing
                processor.alignment_engine = InferenceEngine.create_engine(engine_type, actual_model)

                # Process alignment text with selected model
                alignment_response = processor.process_alignment(alignment_text, params)
                
                # Store in session state for display and copy functionality
                st.session_state.alignment_response = alignment_response
                
                # Force a rerun to update the display
                st.rerun()

            except Exception as e:
                st.error(f"Error in alignment: {str(e)}")
                st.session_state.alignment_response = f"**Error:** {str(e)}"

    # Right column - Main prompt section
    with right_col:
        st.header("Main Prompt")

        # Main prompt input
        prompt = st.text_area(
            "Enter your prompt here",
            placeholder="Enter your prompt...",
            height=150
        )
        
        # Add some spacing to align with the left column's layout
        st.write("")
        st.write("")

        # Process button
        if st.button("Process", type="primary"):
            if prompt:
                try:
                    # Use custom model name if checkbox is checked
                    actual_main_model = main_custom_model if main_use_custom_model and main_custom_model else main_model
                    st.info(f"Using main model: {actual_main_model} with engine: {main_engine_type}")

                    # Set the main engine based on selected engine and model
                    processor.set_main_engine(main_engine_type, actual_main_model)

                    # Combine alignment and prompt
                    full_prompt = f"{alignment_text.strip()}\n\n{prompt.strip()}" if alignment_text else prompt

                    final_output = processor.process_main(
                        full_prompt,
                        "",
                        params
                    )
                    
                    # Store in session state for the copy functionality
                    st.session_state.final_output = final_output
                    st.session_state.history.append({
                        "prompt": prompt,
                        "response": final_output
                    })
                    
                    # Force a rerun to update the display
                    st.rerun()

                except Exception as e:
                    st.error(f"Error in processing: {str(e)}")
            else:
                st.warning("Please enter a prompt to process")

        # Model response output - always visible
        st.subheader("Model Response")
        model_output_container = st.container(height=200, border=True)
        with model_output_container:
            st.markdown(st.session_state.final_output)

    # Add a copy button for alignment response
    if st.session_state.alignment_response:
        with st.expander("Copy alignment response", expanded=False):
            st.code(st.session_state.alignment_response)

    if 'final_output' in st.session_state and st.session_state.final_output:
        with st.expander("Copy model response", expanded=False):
            st.code(st.session_state.final_output)

    return None  # Streamlit doesn't need to return an interface object
