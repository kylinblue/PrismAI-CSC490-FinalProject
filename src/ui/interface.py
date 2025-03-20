import gradio as gr
from src.preprocessing.processor import PromptProcessor
from src.inferencing.inference import InferenceEngine, OllamaEngine, ClaudeEngine, OpenAIEngine

def create_interface(processor: PromptProcessor) -> gr.Interface:
    def get_params_dict(style, tone, creativity):
        """Extract actual values from Gradio components"""
        return {
            "style": style,
            "tone": tone,
            "creativity": float(creativity),
            "format": "json"  # Always use json format
        }
    def get_models_for_engine(engine_type: str) -> list:
        """Get available models for the selected engine type"""
        try:
            if engine_type == "ollama":
                models = OllamaEngine.get_available_models()
                if not models:
                    print("Warning: No Ollama models found. Is Ollama running?")
                    return ["(none available - is Ollama running?)"]
                print(f"Found Ollama models: {models}")
                return models
            elif engine_type == "claude":
                return ClaudeEngine.get_available_models()
            elif engine_type == "openai":
                return OpenAIEngine.get_available_models()
            else:
                return ["(unknown engine)"]
        except Exception as e:
            print(f"Error getting models for {engine_type}: {str(e)}")
            return ["(error loading models)"]
    
    def update_models_dropdown(engine_type: str):
        """Update the models dropdown based on selected engine"""
        return gr.Dropdown(choices=get_models_for_engine(engine_type))
    
    def process_alignment_only(engine_type: str, model: str, use_custom_model: bool, custom_model: str,
                              params: dict, alignment_text: str) -> str:
        """Process only the alignment text with the alignment model"""
        try:
            # Use custom model name if checkbox is checked
            actual_model = custom_model if use_custom_model and custom_model else model
            print(f"Using alignment model: {actual_model} with engine: {engine_type}")
            
            # Set the alignment engine explicitly before processing
            processor.alignment_engine = InferenceEngine.create_engine(engine_type, actual_model)
            
            # Process alignment text with selected model
            alignment_response = processor.process_alignment(alignment_text, params)
            
            return alignment_response
        except Exception as e:
            error_message = f"Error processing alignment: {str(e)}"
            print(error_message)
            return f"Error in alignment: {error_message}"
    
    def process_prompt(alignment_engine_type: str, alignment_model: str, alignment_use_custom: bool, alignment_custom_model: str,
                      main_engine_type: str, main_model: str, main_use_custom: bool, main_custom_model: str,
                      params: dict, alignment_text: str, alignment_response: str, prompt: str) -> str:
        try:
            # Use custom model names if checkbox is checked
            actual_main_model = main_custom_model if main_use_custom and main_custom_model else main_model
            print(f"Using main model: {actual_main_model} with engine: {main_engine_type}")
            
            # Set the main engine based on selected engine and model
            processor.set_main_engine(main_engine_type, actual_main_model)
            
            # Process main prompt with selected model, using alignment results
            final_output = processor.process_main(prompt, alignment_response, params)
            
            return final_output
        except Exception as e:
            error_message = f"Error processing prompt: {str(e)}"
            print(error_message)
            return f"Error in processing: {error_message}"
    
    def process_full_pipeline(alignment_engine_type: str, alignment_model: str, alignment_use_custom: bool, alignment_custom_model: str,
                             main_engine_type: str, main_model: str, main_use_custom: bool, main_custom_model: str,
                             alignment_text: str, prompt: str) -> tuple[str, str]:
        """Process both alignment and main prompt in one go"""
        try:
            # Create params dictionary with actual values from the outer scope
            style_value = params["style"].value if hasattr(params["style"], "value") else params["style"]
            tone_value = params["tone"].value if hasattr(params["tone"], "value") else params["tone"]
            creativity_value = params["creativity"].value if hasattr(params["creativity"], "value") else params["creativity"]
            
            process_params = get_params_dict(style_value, tone_value, creativity_value)
            
            # Process alignment first
            alignment_response = process_alignment_only(
                alignment_engine_type, alignment_model, alignment_use_custom, alignment_custom_model,
                process_params, alignment_text
            )
            
            # Then process main prompt
            final_output = process_prompt(
                alignment_engine_type, alignment_model, alignment_use_custom, alignment_custom_model,
                main_engine_type, main_model, main_use_custom, main_custom_model,
                process_params, alignment_text, alignment_response, prompt
            )
            
            return alignment_response, final_output
        except Exception as e:
            error_message = f"Error in processing pipeline: {str(e)}"
            print(error_message)
            return f"Error in alignment: {error_message}", f"Error in processing: {error_message}"

    available_engines = InferenceEngine.get_available_engines()
    initial_engine = available_engines[0] if available_engines else "placeholder"
    initial_models = get_models_for_engine(initial_engine)

    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column(scale=4):
                engine_dropdown = gr.Dropdown(
                    choices=available_engines,
                    label="Available Engines",
                    value=initial_engine
                )
                
                model_dropdown = gr.Dropdown(
                    choices=initial_models,
                    label="Available weak models",
                    value=initial_models[0] if initial_models else None
                )
            
            with gr.Column(scale=1):
                align_refresh_btn = gr.Button("Refresh Models", variant="secondary")
            
            with gr.Column(scale=2):
                custom_model_checkbox = gr.Checkbox(
                    label="Use custom model name",
                    value=False
                )
                
                custom_model_input = gr.Textbox(
                    label="Custom model name",
                    visible=False
                )
            
            # Update models when engine changes
            engine_dropdown.change(
                fn=update_models_dropdown,
                inputs=engine_dropdown,
                outputs=model_dropdown
            )
            
            # Toggle custom model input visibility
            custom_model_checkbox.change(
                fn=lambda x: gr.update(visible=x),
                inputs=custom_model_checkbox,
                outputs=custom_model_input
            )

        with gr.Accordion("Processing Parameters", open=False):
            params = {
                "style": gr.Radio(
                    choices=["Casual", "Professional", "Academic"],
                    label="Writing Style",
                    value="Professional"
                ),
                "tone": gr.Dropdown(
                    choices=["Neutral", "Positive", "Negative"],
                    label="Tone",
                    value="Neutral"
                ),
                "creativity": gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    label="Creativity Level"
                )
            }

        with gr.Row():
            with gr.Column():
                alignment_input = gr.Textbox(
                    label="Add your alignment text here",
                    placeholder="Enter text to align the model's behavior...",
                    lines=4)
                website_checkbox = gr.Checkbox(
                    label="Website",
                    value=False
                )
                
                # Add alignment-specific buttons
                with gr.Row():
                    align_btn = gr.Button("Align", variant="primary")
                    align_interrupt_btn = gr.Button("Interrupt", variant="stop")
                    align_reset_btn = gr.Button("Reset", variant="secondary")
            
        with gr.Row():
            alignment_output = gr.Textbox(
                label="Model response to alignment",
                lines=2,
                interactive=False
            )

        # Add second engine selection for main model
        with gr.Row():
            with gr.Column(scale=4):
                main_engine_dropdown = gr.Dropdown(
                    choices=available_engines,
                    label="Main Processing Engine",
                    value=initial_engine
                )
                
                main_model_dropdown = gr.Dropdown(
                    choices=initial_models,
                    label="Main Processing Model",
                    value=initial_models[0] if initial_models else None
                )
            
            with gr.Column(scale=1):
                main_refresh_btn = gr.Button("Refresh Models", variant="secondary")
            
            with gr.Column(scale=2):
                main_custom_model_checkbox = gr.Checkbox(
                    label="Use custom main model name",
                    value=False
                )
                
                main_custom_model_input = gr.Textbox(
                    label="Custom main model name",
                    visible=False
                )
            
            # Update models when engine changes
            main_engine_dropdown.change(
                fn=update_models_dropdown,
                inputs=main_engine_dropdown,
                outputs=main_model_dropdown
            )
            
            # Toggle custom model input visibility
            main_custom_model_checkbox.change(
                fn=lambda x: gr.update(visible=x),
                inputs=main_custom_model_checkbox,
                outputs=main_custom_model_input
            )
            
            # Add refresh button functionality for alignment model
            align_refresh_btn.click(
                fn=update_models_dropdown,
                inputs=engine_dropdown,
                outputs=model_dropdown
            )
            
            # Add refresh button functionality for main model
            main_refresh_btn.click(
                fn=update_models_dropdown,
                inputs=main_engine_dropdown,
                outputs=main_model_dropdown
            )

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="Enter your main prompt here...",
                lines=6
            )

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            interrupt_btn = gr.Button("Interrupt", variant="stop")

        with gr.Row():
            output = gr.Textbox(
                label="Model output",
                lines=5,
                interactive=False
            )

        # Add review message
        gr.Markdown("""
        ### How to use:
        1. Enter your alignment text and click Submit to see the alignment interpretation
        2. Review the alignment response
        3. Enter your main prompt and click Submit again to get the final response
        
        Note: The system will optimize your prompt based on the alignment principles before sending it to the main model.
        """)

        # Add functionality for alignment buttons
        align_btn.click(
            fn=process_alignment_only,
            inputs=[
                engine_dropdown,  # Use the alignment engine dropdown
                model_dropdown,   # Use the alignment model dropdown
                custom_model_checkbox,
                custom_model_input,
                gr.JSON(lambda: get_params_dict(params["style"].value, params["tone"].value, params["creativity"].value)),
                alignment_input
            ],
            outputs=alignment_output
        )
        
        # Reset button for alignment
        align_reset_btn.click(
            fn=lambda: "",
            outputs=alignment_output
        )
        
        # Create a function to process only the main prompt using existing alignment
        def process_main_only(main_engine_type, main_model, main_use_custom, main_custom_model,
                             params_dict, alignment_result, prompt):
            try:
                # Use custom model name if checkbox is checked
                actual_main_model = main_custom_model if main_use_custom and main_custom_model else main_model
                print(f"Using main model: {actual_main_model} with engine: {main_engine_type}")
            
                # Set the main engine based on selected engine and model
                processor.set_main_engine(main_engine_type, actual_main_model)
            
                # Process main prompt with selected model, using existing alignment results
                final_output = processor.process_main(prompt, alignment_result, params_dict)
            
                return final_output
            except Exception as e:
                error_message = f"Error processing main prompt: {str(e)}"
                print(error_message)
                return f"Error in processing: {error_message}"
        
        # Main submit button for full pipeline (only when no alignment has been done yet)
        def conditional_process(alignment_output_text, *args):
            """Only process what's needed based on current state"""
            if not alignment_output_text.strip():
                # If no alignment has been done yet, process both
                return process_full_pipeline(*args)
            else:
                # If alignment is already done, only process main prompt and return None for alignment
                # to avoid updating the alignment output box
                main_result = process_main_only(
                    args[4], args[5], args[6], args[7],
                    get_params_dict(params["style"].value, params["tone"].value, params["creativity"].value),
                    alignment_output_text, args[9]
                )
                return gr.update(value=alignment_output_text), main_result

        submit_btn.click(
            fn=conditional_process,
            inputs=[
                alignment_output,  # Check if alignment is already done
                engine_dropdown,
                model_dropdown,
                custom_model_checkbox,
                custom_model_input,
                main_engine_dropdown,
                main_model_dropdown,
                main_custom_model_checkbox,
                main_custom_model_input,
                alignment_input,
                prompt_input
            ],
            outputs=[alignment_output, output]
        )

    return interface
