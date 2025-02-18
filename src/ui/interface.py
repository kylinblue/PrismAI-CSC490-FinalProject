import gradio as gr
from src.preprocessing.processor import PromptProcessor

def create_interface(processor: PromptProcessor) -> gr.Interface:
    def process_prompt(input_text: str, params: dict) -> str:
        return processor.process(input_text, params)

    return gr.Interface(
        fn=process_prompt,
        inputs=[
            gr.Textbox(label="Enter your prompt"),
            gr.JSON(label="Processing Parameters")
        ],
        outputs=gr.Textbox(label="Modified Prompt")
    )