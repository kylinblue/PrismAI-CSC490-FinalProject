import gradio as gr
from src.preprocessing.processor import PromptProcessor
from src.inferencing.inference import InferenceEngine, OllamaEngine, ClaudeEngine, OpenAIEngine

def create_interface(processor: PromptProcessor) -> gr.Interface:
    def get_models_for_engine(engine_type: str) -> list:
        """Get available models for the selected engine type"""
        try:
            if engine_type == "ollama":
                models = OllamaEngine.get_available_models()
                return models if models else ["(none available)"]
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
    
    def process_prompt(engine_type: str, model: str, params: dict, alignment_text: str, prompt: str) -> tuple[str, str]:
        # Set the main engine based on selected engine and model
        processor.set_main_engine(engine_type, model)
        
        # First process alignment text with Llama
        alignment_response = processor.process_alignment(alignment_text, params)
        
        # Then process main prompt with selected model, using alignment results
        final_output = processor.process_main(prompt, alignment_response, params)
        
        return alignment_response, final_output

    available_engines = InferenceEngine.get_available_engines()
    initial_engine = available_engines[0] if available_engines else "placeholder"
    initial_models = get_models_for_engine(initial_engine)

    with gr.Blocks() as interface:
        with gr.Row():
            engine_dropdown = gr.Dropdown(
                choices=available_engines,
                label="Available Engines",
                value=initial_engine
            )
            
            model_dropdown = gr.Dropdown(
                choices=initial_models,
                label="Available models",
                value=initial_models[0] if initial_models else None
            )
            
            # Update models when engine changes
            engine_dropdown.change(
                fn=update_models_dropdown,
                inputs=engine_dropdown,
                outputs=model_dropdown
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
            
        with gr.Row():
            alignment_output = gr.Textbox(
                label="Model response to alignment",
                lines=2,
                interactive=False
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
            refresh_btn = gr.Button("Refresh Models", variant="secondary")
            
            # Add refresh button functionality
            refresh_btn.click(
                fn=update_models_dropdown,
                inputs=engine_dropdown,
                outputs=model_dropdown
            )

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
        """)

        with gr.Row():
            submit_btn.click(
                fn=process_prompt,
                inputs=[
                    engine_dropdown,
                    model_dropdown,
                    gr.JSON(value=params),
                    alignment_input,
                    prompt_input
                ],
                outputs=[alignment_output, output]
            )

    return interface
