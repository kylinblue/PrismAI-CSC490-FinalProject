import gradio as gr
from src.preprocessing.processor import PromptProcessor

def create_interface(processor: PromptProcessor) -> gr.Interface:
    def process_prompt(model: str, params: dict, alignment_text: str, prompt: str) -> tuple[str, str]:
        # Set the main engine based on selected model
        processor.set_main_engine("claude" if model == "Claude" else "ollama", model)
        
        # First process alignment text with Llama
        alignment_response = processor.process_alignment(alignment_text, params)
        
        # Then process main prompt with selected model, using alignment results
        final_output = processor.process_main(prompt, alignment_response, params)
        
        return alignment_response, final_output

    available_models = ["GPT-4", "Claude", "Mistral", "Llama", "Placeholder"]

    with gr.Blocks() as interface:
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                label="Available models",
                value=available_models[0]
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
                    model_dropdown,
                    gr.JSON(value=params),
                    alignment_input,
                    prompt_input
                ],
                outputs=[alignment_output, output]
            )

    return interface
